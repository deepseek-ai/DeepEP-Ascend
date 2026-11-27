import warnings
warnings.filterwarnings('ignore', message='Permission mismatch.*', module='torch_npu.utils._path_manager')
warnings.filterwarnings('ignore', message=r'Warning: The .* owner does not match the current owner\.', module='torch_npu.utils.collect_env')

import argparse
import json
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, bench_msprof, generate_lognormal_topk, parse_num_bytes

# Period of `make_pattern`'s ramp, chosen so adjacent payload chunks have different contents
NUM_PATTERN_VALUES = (1 << 12) - 1

# Hidden sizes in fp32 elements, including full chunks, partial tails, and buffer reuse
CORRECTNESS_HIDDENS = (1 << 20, 1 << 18, 16392, 6144, 3080, 3072, 3064, 5120, 1536, 384, 24)


def make_pattern(hidden: int, key: int) -> torch.Tensor:
    """A deterministic fp32 blob: a ramp of period `NUM_PATTERN_VALUES` shifted by `key`.

    Every blob in the test carries a distinct key, so a swapped source shows up as a
    mismatch at every element. Adjacent payload chunks also have different contents, making
    a one-chunk offset error visible.

    The values are small integers on purpose. `lb_reduce_grads` sums redundant expert gradients into
    `expert_grads` with element-wise accumulation (URMA READ_REDUCE for remote contributions), so the order in
    which several borrowers land is not fixed. Values this size stay exact in fp32, which
    makes the expected value order-independent and lets the check use `torch.equal` rather
    than a tolerance.

    Plain arithmetic rather than an RNG: it must come out bit-identical on every NPU,
    and RNG launch configs depend on device properties.
    """
    idx = torch.arange(hidden, dtype=torch.int64, device='npu') + key
    return ((idx % NUM_PATTERN_VALUES) - NUM_PATTERN_VALUES // 2).to(torch.float32)


def expert_grad_key(expert_idx: int) -> int:
    """Key of an original expert gradient. Non-negative, while `redundant_expert_grad_key` is negative, so the two
    families of blobs never share a key."""
    return expert_idx


def redundant_expert_grad_key(rank_idx: int, redundant_expert_idx: int, num_redundant_experts_per_rank: int) -> int:
    """Key of a redundant expert gradient, unique over `(rank_idx, redundant_expert_idx)`."""
    return -1 - (rank_idx * num_redundant_experts_per_rank + redundant_expert_idx)


def expected_expert_grads(redundancy_mapping: torch.Tensor, rank_idx: int, num_ranks: int,
                          num_redundant_experts_per_rank: int, num_main_experts_per_rank: int, hidden: int) -> torch.Tensor:
    """The seed gradients plus the contributions from every matching redundant expert."""
    expected = torch.stack([make_pattern(hidden, expert_grad_key(rank_idx * num_main_experts_per_rank + local_expert_idx))
                            for local_expert_idx in range(num_main_experts_per_rank)])
    for remote_rank_idx in range(num_ranks):
        for redundant_expert_idx in range(num_redundant_experts_per_rank):
            expert_idx = int(redundancy_mapping[remote_rank_idx, redundant_expert_idx].item())
            if expert_idx >= 0 and expert_idx // num_main_experts_per_rank == rank_idx:
                expected[expert_idx % num_main_experts_per_rank] += \
                    make_pattern(hidden, redundant_expert_grad_key(remote_rank_idx, redundant_expert_idx, num_redundant_experts_per_rank))
    return expected


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2 and args.num_experts > 0 and args.num_experts % num_ranks == 0
    num_main_experts_per_rank = args.num_experts // num_ranks
    # `lb_get_redundancy_mapping` hands back one copy slot per local expert, so that bounds the redundant pool
    num_redundant_experts_per_rank = num_main_experts_per_rank
    assert args.num_experts + num_ranks * num_redundant_experts_per_rank < NUM_PATTERN_VALUES
    assert args.num_hidden_bytes % deep_ep.get_num_tma_alignment() == 0
    hiddens = [*CORRECTNESS_HIDDENS, args.num_hidden_bytes // 4]
    plan = deep_ep.BufferAllocator()
    pools = [plan.allocate((num_redundant_experts_per_rank, h), torch.float32) for h in hiddens]
    buffer = deep_ep.EPBuffer(group, num_bytes=deep_ep.get_num_allocation_alignment(),
                              lb_allocation_plan_or_num_bytes=plan,
                              explicitly_destroy=True)
    num_sms = args.num_sms
    outputs = [torch.empty((num_main_experts_per_rank, h), dtype=torch.float32, device='npu') for h in hiddens]

    def current_mapping(round_idx: int = 0) -> torch.Tensor:
        """The redundancy mapping this round runs with: `lb_get_redundancy_mapping`'s plan for a lognormal load."""
        topk_idx = generate_lognormal_topk(args.num_tokens, args.num_experts, args.num_topk, rank_idx,
                                       args.seed + round_idx)
        _, redundancy_mapping = buffer.lb_get_redundancy_mapping(topk_idx, args.num_experts, num_redundant_experts_per_rank, args.token_alignment, 0)
        return redundancy_mapping

    def run(src, dst, redundancy_mapping):
        buffer.lb_reduce_grads(src, dst, redundancy_mapping, num_sms=num_sms).wait()

    def seed(src, dst, hidden):
        for e in range(num_main_experts_per_rank):
            dst[e] = make_pattern(hidden, expert_grad_key(rank_idx * num_main_experts_per_rank + e))
        for e in range(num_redundant_experts_per_rank):
            src[e] = make_pattern(hidden, redundant_expert_grad_key(rank_idx, e, num_redundant_experts_per_rank))

    for hidden, src, dst in zip(CORRECTNESS_HIDDENS, pools, outputs):
        for round_idx in range(args.num_rounds):
            redundancy_mapping = current_mapping(round_idx)
            seed(src, dst, hidden)
            original = dst.clone()
            src_before = src.clone()
            expected = expected_expert_grads(redundancy_mapping, rank_idx, num_ranks,
                                            num_redundant_experts_per_rank, num_main_experts_per_rank, hidden)
            if hidden == CORRECTNESS_HIDDENS[0] and round_idx == 0 and src.numel() > 0:
                # Remote READ must reject ordinary, unregistered source tensors before launch.
                try:
                    run(src.clone(), dst, redundancy_mapping)
                except RuntimeError as error:
                    assert 'ptr >= begin' in str(error), str(error)
                else:
                    raise AssertionError('Accepted a redundant source outside the LB region')
            run(src, dst, redundancy_mapping)
            if not torch.equal(dst, expected):
                bad = (dst != expected).nonzero()
                sample = bad[:16]
                print(f'REDUCE_MISMATCH rank={rank_idx} hidden={hidden} count={bad.size(0)} '
                      f'indices={sample.cpu().tolist()} actual={dst[sample[:, 0], sample[:, 1]].cpu().tolist()} '
                      f'expected={expected[sample[:, 0], sample[:, 1]].cpu().tolist()}', flush=True)
                raise AssertionError(f'{hidden=}: reduction mismatch')
            # Reuse the queues and add a second contribution without resetting the destination.
            run(src, dst, redundancy_mapping)
            repeated_expected = expected + (expected - original)
            if not torch.equal(dst, repeated_expected):
                bad = (dst != repeated_expected).nonzero()
                sample = bad[:16]
                print(f'REDUCE_REPEAT_MISMATCH rank={rank_idx} hidden={hidden} count={bad.size(0)} '
                      f'indices={sample.cpu().tolist()} actual={dst[sample[:, 0], sample[:, 1]].cpu().tolist()} '
                      f'expected={repeated_expected[sample[:, 0], sample[:, 1]].cpu().tolist()}', flush=True)
                raise AssertionError(f'{hidden=}: repeated reduction mismatch')
            assert torch.equal(src, src_before), f'{hidden=}: source was modified'
        dist_print(f'Correctness passed: hidden={hidden}', once_in_node=True)

    # WRITE and READ share the context's SQ/CQ counters across kernel invocations.
    src, dst = pools[len(CORRECTNESS_HIDDENS) - 1], outputs[len(CORRECTNESS_HIDDENS) - 1]
    hidden = CORRECTNESS_HIDDENS[-1]
    redundancy_mapping = current_mapping()
    for _ in range(2):
        seed(src, dst, hidden)
        buffer.lb_prefetch_weights(src, dst, redundancy_mapping, num_sms=num_sms).wait()
        seed(src, dst, hidden)
        expected = expected_expert_grads(redundancy_mapping, rank_idx, num_ranks,
                                        num_redundant_experts_per_rank, num_main_experts_per_rank, hidden)
        run(src, dst, redundancy_mapping)
        assert torch.equal(dst, expected), 'WRITE/READ CQ reuse mismatch'
    dist_print('Mixed prefetch/reduce CQ reuse passed', once_in_node=True)

    actual_sms = buffer.lb_get_theoretical_num_sms() if num_sms == 0 else num_sms
    src, dst = pools[-1], outputs[-1]
    redundancy_mapping = current_mapping()
    redundancy_mapping_cpu = redundancy_mapping.cpu()
    tx, rx = [0] * num_ranks, [0] * num_ranks
    src.fill_(rank_idx + 1)
    contribution = torch.zeros(num_main_experts_per_rank, dtype=torch.float32, device='cpu')
    for borrower in range(num_ranks):
        for expert_idx in redundancy_mapping_cpu[borrower].tolist():
            if expert_idx < 0:
                continue
            owner = expert_idx // num_main_experts_per_rank
            if owner != borrower:
                tx[borrower] += 1
                rx[owner] += 1
            if owner == rank_idx:
                contribution[expert_idx % num_main_experts_per_rank] += borrower + 1
    contribution = contribution.npu().unsqueeze(1)
    for repeat in range(args.num_repeats):
        dst.fill_(1)
        run(src, dst, redundancy_mapping)
        assert (dst == 1 + contribution).all(), 'Performance payload mismatch'
        dst.fill_(1)
        t, _, _ = bench(lambda: run(src, dst, redundancy_mapping), num_warmups=args.num_warmups, num_tests=args.num_tests,
                        barrier=lambda: dist.barrier(group))
        assert (dst == 1 + (args.num_warmups + args.num_tests) * contribution).all(), \
            'Repeated performance payload mismatch'
        assert (src == rank_idx + 1).all()
        duration = torch.tensor([t], dtype=torch.float32, device='npu')
        dist.all_reduce(duration, op=dist.ReduceOp.MAX, group=group)
        t = duration.item()
        row = dict(seed=args.seed,
                   repeat=repeat, num_ranks=num_ranks, num_sms=actual_sms,
                   num_hidden_bytes=args.num_hidden_bytes, tx_counts=tx, rx_counts=rx, us=t * 1e6,
                   gbs=max(tx + rx) * args.num_hidden_bytes / t / 1e9)
        dist_print(f'PERF {json.dumps(row)}', once_in_node=True)
    dst.fill_(1)
    names = ['lb_reduce_grads_issue', 'barrier']
    profile = bench_msprof(lambda: run(src, dst, redundancy_mapping), kernel_names=names,
                           num_warmups=3, num_tests=10, flush_l2=False,
                           barrier_comm_profiling=True, barrier=lambda: dist.barrier(group))
    kernel_times = dict(issue_us=profile[0].us, epilogue_us=profile[1].us)
    profile = profile[0] + profile[1]
    assert (dst == 1 + 13 * contribution).all(), 'Profiler payload mismatch'
    assert (src == rank_idx + 1).all()
    sample = dict(rank=rank_idx, kernel_us=profile.us,
                  aiv_scalar=profile.aiv_scalar, aiv_mte2=profile.aiv_mte2, aiv_mte3=profile.aiv_mte3,
                  aiv_vec=profile.aiv_vec, **kernel_times)
    profiles = [None] * num_ranks
    dist.all_gather_object(profiles, sample, group=group)
    if rank_idx == 0:
        print(f'PROFILE {json.dumps(profiles)}', flush=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Ascend expert gradient reduction')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-hidden-bytes', type=parse_num_bytes, default=64 << 20)
    parser.add_argument('--num-rounds', type=int, default=2)
    parser.add_argument('--num-tokens', type=int, default=8192,
                        help='Tokens per rank fed to lb_get_redundancy_mapping')
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--token-alignment', type=int, default=128)
    parser.add_argument('--num-sms', type=int, default=0, help='AI core count; 0 uses lb_get_theoretical_num_sms()')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-warmups', type=int, default=5)
    parser.add_argument('--num-tests', type=int, default=15)
    parser.add_argument('--num-repeats', type=int, default=1)
    args = parser.parse_args()
    if args.num_repeats <= 0 or args.num_tests <= 0 or args.num_warmups < 0:
        parser.error('repeats/tests must be positive; warmups must be nonnegative')
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
