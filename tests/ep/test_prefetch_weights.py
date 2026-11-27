import argparse
import json
import statistics

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, bench_msprof, generate_lognormal_topk, parse_num_bytes

# Written into every redundant expert slot before a push. Slots whose expert index is `-1`
# must still hold it afterwards, which also catches strayed writes.
SENTINEL = 0xa5

# The three payloads an expert pushes, each checked on its own
PAYLOAD_WEIGHTS, PAYLOAD_SF0, PAYLOAD_SF1 = range(3)
PAYLOAD_NAMES = ('weights', 'sf0', 'sf1')
NUM_PAYLOADS = len(PAYLOAD_NAMES)

# The ramp distinguishes expert/payload identities and adjacent WRITE chunks.
NUM_BLOB_VALUES = (1 << 8) - 1

# Payload sizes the correctness sweep covers, in bytes: `(weights, sf0, sf1)` per expert.
PAYLOAD_SWEEP = (
    (4 << 20, 1 << 14, 1 << 13),  # Large correctness payloads
    (1 << 20, 1 << 14, 1 << 12),  # Large weights with shorter SF payloads
    (49152, 24576, 12288),        # Different payload lengths
    (12320, 12256, 32),           # Sizes around a 12 KiB boundary
    (24640, 12320, 64),           # Different non-power-of-two lengths
    (49152, 3072, 1024),          # Weights with shorter SF payloads
    (20480, 20480, 20480),        # Equal-sized payloads
    (1152, 544, 288),             # Small aligned payloads
    (96, 64, 32),                 # Tiny payloads with idle AIVs
    (64, 32, 32),                 # Minimum-size SF payloads
)


def reshape_weights(weights: list[torch.Tensor], trailing_shape: tuple[int, ...]) -> list[torch.Tensor]:
    return [tensor.unflatten(1, (-1, *trailing_shape)) for tensor in weights]


def make_blob(global_expert_idx: int, num_bytes: int, payload_idx: int) -> torch.Tensor:
    """One payload of one expert: a ramp of period `NUM_BLOB_VALUES`, shifted by a key
    unique to `(global_expert_idx, payload_idx)`.

    A blob pushed from the wrong expert, or into the wrong payload, therefore differs at
    every byte from the expected one rather than at a lucky few.

    Plain arithmetic rather than an RNG: it must come out bit-identical on every NPU,
    and RNG launch configs depend on device properties.
    """
    key = global_expert_idx * NUM_PAYLOADS + payload_idx
    idx = torch.arange(num_bytes, dtype=torch.int64, device='npu') + key
    return (idx % NUM_BLOB_VALUES).to(torch.uint8)


def check(redundant_expert_weights: torch.Tensor, redundancy_mapping: torch.Tensor,
          rank_idx: int, num_bytes: int, payload_idx: int, where: str) -> None:
    """Every redundant expert slot of this rank must hold its expert's blob, or the sentinel if unassigned."""
    name = PAYLOAD_NAMES[payload_idx]
    for redundant_expert_idx in range(redundant_expert_weights.size(0)):
        expert_idx = int(redundancy_mapping[rank_idx, redundant_expert_idx].item())
        if expert_idx < 0:
            assert (redundant_expert_weights[redundant_expert_idx] == SENTINEL).all(), \
                f'{where}: {name} redundant expert {redundant_expert_idx} is unassigned but was written'
        else:
            assert torch.equal(redundant_expert_weights[redundant_expert_idx], make_blob(expert_idx, num_bytes, payload_idx)), \
                f'{where}: {name} redundant expert {redundant_expert_idx} does not match expert {expert_idx}'


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2

    # Settings
    num_experts = args.num_experts
    assert num_experts % num_ranks == 0
    num_main_experts_per_rank = num_experts // num_ranks
    num_redundant_experts_per_rank, num_sms = num_main_experts_per_rank, args.num_sms

    # Correctness sweeps a table of payload sizes.
    perf_payload = (args.num_hidden_bytes, args.num_sf0_bytes, args.num_sf1_bytes)
    payloads = (*PAYLOAD_SWEEP, perf_payload)

    assert all(n % deep_ep.get_num_tma_alignment() == 0 for payload in payloads for n in payload), \
        'Each expert payload must satisfy the LB API alignment'
    assert num_experts * NUM_PAYLOADS <= NUM_BLOB_VALUES, \
        'Two blobs would share a key modulo the pattern period'

    # Plan every symmetric redundant expert pool once, including the benchmark pools.
    plan = deep_ep.BufferAllocator()
    redundant_expert_weight_sets = [
        [plan.allocate((num_redundant_experts_per_rank, n), torch.uint8) for n in payload]
        for payload in payloads
    ]
    buffer = deep_ep.EPBuffer(group, num_bytes=deep_ep.get_num_allocation_alignment(),
                              lb_allocation_plan_or_num_bytes=plan, explicitly_destroy=True)
    num_sms = buffer.lb_get_theoretical_num_sms() if num_sms == 0 else num_sms

    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Experts: {num_experts} ({num_main_experts_per_rank} local), '
               f'redundant experts: {num_redundant_experts_per_rank}\n'
               f' > Correctness payload sweep: {[list(payload) for payload in PAYLOAD_SWEEP]} B\n'
               f' > Perf bytes per expert: {perf_payload[PAYLOAD_WEIGHTS]} weights, '
               f'{perf_payload[PAYLOAD_SF0]}/{perf_payload[PAYLOAD_SF1]} SF '
               f'({sum(perf_payload) / 1e6:.1f} MB total)\n'
               f' > #AI cores: {num_sms}\n',
               once_in_node=True)

    # This rank's original expert weights, one set per payload size.
    expert_weight_sets = [
        [torch.empty((num_main_experts_per_rank, n), dtype=torch.uint8, device='npu') for n in payload]
        for payload in payloads
    ]

    sweep_expert_weight_sets = expert_weight_sets[:len(PAYLOAD_SWEEP)]
    sweep_redundant_expert_weight_sets = redundant_expert_weight_sets[:len(PAYLOAD_SWEEP)]

    for payload, expert_weights in zip(PAYLOAD_SWEEP, sweep_expert_weight_sets, strict=True):
        for local_expert_idx in range(num_main_experts_per_rank):
            expert_idx = rank_idx * num_main_experts_per_rank + local_expert_idx
            for payload_idx, num_bytes in enumerate(payload):
                expert_weights[payload_idx][local_expert_idx] = make_blob(expert_idx, num_bytes, payload_idx)

    def current_mapping(round_idx: int = 0) -> torch.Tensor:
        """The redundancy mapping this round runs with: `lb_get_redundancy_mapping`'s plan for a lognormal load."""
        topk_idx = generate_lognormal_topk(args.num_tokens, num_experts, args.num_topk, rank_idx, args.seed + round_idx)
        _, redundancy_mapping = buffer.lb_get_redundancy_mapping(topk_idx, num_experts, num_redundant_experts_per_rank, args.token_alignment, 0)
        return redundancy_mapping

    def prefetch(redundant_expert_weights: list, expert_weights: list, redundancy_mapping: torch.Tensor,
                 with_sf: bool, previous_event=None):
        return buffer.lb_prefetch_weights(
            redundant_expert_weights if with_sf else redundant_expert_weights[:1],
            expert_weights if with_sf else expert_weights[:1],
            redundancy_mapping, num_sms=num_sms, previous_event=previous_event)

    # Check correctness over every payload size.
    dist_print(f'Checking correctness over {len(PAYLOAD_SWEEP)} payload sizes '
               f'x {args.num_rounds} rounds ...', once_in_node=True)
    for payload, redundant_expert_weights, expert_weights in zip(PAYLOAD_SWEEP, sweep_redundant_expert_weight_sets, sweep_expert_weight_sets, strict=True):
        for with_sf in (True, False):
            for round_idx in range(args.num_rounds):
                where = f'{payload}, {with_sf=}, {round_idx=}'
                sweep_redundancy_mapping = current_mapping(round_idx)

                # The sentinel fill must be visible to peers before anyone pushes
                for tensor in redundant_expert_weights:
                    tensor.fill_(SENTINEL)
                torch.npu.synchronize()
                dist.barrier(group)

                # Alternate flat tensors with 3D/4D views; the expert byte payloads stay the same.
                prefetch(
                    reshape_weights(redundant_expert_weights, (2, 8)) if round_idx % 2 else redundant_expert_weights,
                    reshape_weights(expert_weights, (16,)) if round_idx % 2 else expert_weights,
                    sweep_redundancy_mapping, with_sf, previous_event=buffer.capture() if round_idx % 2 else None).wait()
                torch.npu.synchronize()

                num_pushed = NUM_PAYLOADS if with_sf else 1
                for payload_idx in range(num_pushed):
                    check(redundant_expert_weights[payload_idx], sweep_redundancy_mapping, rank_idx,
                          payload[payload_idx], payload_idx, where)
                for payload_idx in range(num_pushed, NUM_PAYLOADS):
                    assert (redundant_expert_weights[payload_idx] == SENTINEL).all(), \
                        f'{where}: {PAYLOAD_NAMES[payload_idx]} was pushed even though it was not requested'

    # The planned mapping leaves empty slots whenever the plan needs no copy there
    payload = PAYLOAD_SWEEP[-1]
    redundant_expert_weights = sweep_redundant_expert_weight_sets[-1]
    expert_weights = sweep_expert_weight_sets[-1]
    redundancy_mapping = current_mapping()
    for tensor in redundant_expert_weights:
        tensor.fill_(SENTINEL)
    prefetch(redundant_expert_weights, expert_weights, redundancy_mapping, True).wait()
    torch.npu.synchronize()
    for payload_idx, num_bytes in enumerate(payload):
        check(redundant_expert_weights[payload_idx], redundancy_mapping, rank_idx, num_bytes, payload_idx, 'empty slots')

    if num_redundant_experts_per_rank > 0:
        # Reject ordinary destination tensors before constructing an invalid remote address.
        try:
            prefetch([torch.empty_like(tensor) for tensor in redundant_expert_weights], expert_weights,
                     redundancy_mapping, True).wait()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Non-symmetric destinations must be rejected')

    # A changed source must be read again on repeated calls, without a host-side barrier. Row `rank_idx`
    # of the mapping names the expert each of this rank's copy slots mirrors, and -1 marks an empty slot.
    redundancy_mapping = current_mapping()
    redundancy_mapping_cpu = redundancy_mapping.cpu()
    for value in (17, 93, 201):
        for tensor in expert_weights:
            tensor.fill_(value + rank_idx)
        prefetch(redundant_expert_weights, expert_weights, redundancy_mapping, True).wait()
        for redundant_expert_idx in range(num_redundant_experts_per_rank):
            expert_idx = int(redundancy_mapping_cpu[rank_idx, redundant_expert_idx].item())
            if expert_idx < 0:
                continue
            expected = value + expert_idx // num_main_experts_per_rank
            for tensor in redundant_expert_weights:
                assert (tensor[redundant_expert_idx] == expected).all()

    dist_print('All correctness checks passed', once_in_node=True)
    perf_expert_weights = expert_weight_sets[-1]
    perf_redundant_expert_weights = redundant_expert_weight_sets[-1]
    for local_expert_idx in range(num_main_experts_per_rank):
        for payload_idx, num_bytes in enumerate(perf_payload):
            perf_expert_weights[payload_idx][local_expert_idx] = make_blob(
                rank_idx * num_main_experts_per_rank + local_expert_idx, num_bytes, payload_idx)
    redundancy_mapping = current_mapping()
    for tensor in perf_redundant_expert_weights:
        tensor.fill_(SENTINEL)
    run = lambda: prefetch(perf_redundant_expert_weights, perf_expert_weights, redundancy_mapping, True).wait()
    torch.npu.synchronize()
    dist.barrier(group)
    run()
    samples = []
    for _ in range(args.num_repeats):
        t, _, _ = bench(run, barrier=lambda: dist.barrier(group))
        duration = torch.tensor([t], dtype=torch.float32, device='npu')
        dist.all_reduce(duration, op=dist.ReduceOp.MAX, group=group)
        samples.append(duration.item())
    t = statistics.median(samples)
    table = redundancy_mapping.cpu()
    counts, remote_tx, remote_rx = [0] * num_ranks, [0] * num_ranks, [0] * num_ranks
    for borrower, row in enumerate(table.tolist()):
        for expert in row:
            if expert >= 0:
                owner = expert // num_main_experts_per_rank
                counts[owner] += 1
                remote_tx[owner] += owner != borrower
                remote_rx[borrower] += owner != borrower
    result = dict(seed=args.seed,
                  payload_bytes=perf_payload, num_sms=num_sms, samples_us=[x * 1e6 for x in samples],
                  us=t * 1e6, gbs=max(counts) * sum(perf_payload) / t / 1e9,
                  network_gbs=max(max(remote_tx), max(remote_rx)) * sum(perf_payload) / t / 1e9,
                  requests_per_sender=counts, remote_tx=remote_tx, remote_rx=remote_rx)

    profile = bench_msprof(run, kernel_names=['lb_prefetch_weights_issue', 'barrier'],
                           num_warmups=3, num_tests=10, flush_l2=False,
                           barrier_comm_profiling=True, barrier=lambda: dist.barrier(group))
    kernel_times = dict(issue_us=profile[0].us, epilogue_us=profile[1].us)
    profile = profile[0] + profile[1]
    profiles = [None] * num_ranks
    dist.all_gather_object(profiles, dict(rank=rank_idx, kernel_us=profile.us,
                                          aiv_scalar=profile.aiv_scalar, aiv_mte2=profile.aiv_mte2,
                                          aiv_mte3=profile.aiv_mte3, **kernel_times), group=group)
    result['profiles'] = profiles
    for payload_idx, num_bytes in enumerate(perf_payload):
        check(perf_redundant_expert_weights[payload_idx], redundancy_mapping, rank_idx, num_bytes, payload_idx, 'perf')
    if rank_idx == 0:
        print(json.dumps(result), flush=True)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test weight prefetch')
    parser.add_argument('--num-repeats', type=int, default=3)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-hidden-bytes', type=parse_num_bytes, default=64 << 20,
                        help='Weight bytes per expert in the benchmark, with an optional binary '
                             'suffix: 64M (default), 512K, or plain bytes. Correctness sweeps '
                             'its own table of sizes instead')
    parser.add_argument('--num-sf0-bytes', type=parse_num_bytes, default=1 << 14,
                        help='First scale-factor payload in the benchmark')
    parser.add_argument('--num-sf1-bytes', type=parse_num_bytes, default=1 << 12,
                        help='Second scale-factor payload in the benchmark')
    parser.add_argument('--num-rounds', type=int, default=2,
                        help='Copy tables per payload size, on top of the size sweep')
    parser.add_argument('--num-tokens', type=int, default=8192,
                        help='Tokens per rank fed to lb_get_redundancy_mapping')
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--token-alignment', type=int, default=128)
    parser.add_argument('--num-sms', type=int, default=0, help='AI core count; 0 uses lb_get_theoretical_num_sms()')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    assert args.num_repeats > 0
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
