# Copyright (c) 2026, Lu Lu

"""Independent CPU reference helpers from Ascend/DeepEP, synchronized at 8ffafeb."""


import random


def make_routes(world, b, k, e, seed):
    """Independent CPU routes: no repeated experts within any token."""
    rng = random.Random(seed)
    result = []
    for _ in range(world):
        rows = []
        for _ in range(b):
            chosen = rng.sample(range(e), k)
            rows.append(chosen)
        result.append(rows)
    return result


def expert_entries(routes, experts, world, rank):
    local = experts // world
    result = [[] for _ in range(local)]
    for source, rows in enumerate(routes):
        for token, selected in enumerate(rows):
            for expert in selected:
                if rank * local <= expert < (rank + 1) * local:
                    result[expert - rank * local].append((source, token))
    return result


def payload(sources, tokens, hidden, dtype):
    import torch

    sources, tokens = torch.as_tensor(sources), torch.as_tensor(tokens)
    cols = torch.arange(hidden)
    values = ((sources[:, None] * 3 + tokens[:, None] * 5 + cols) % 17 - 8).to(dtype)
    # Four small exact integers identify every row, even with FP8 input.
    values[:, 0], values[:, 1] = sources // 16, sources % 16
    values[:, 2], values[:, 3] = tokens // 16, tokens % 16
    return values


def fp8_scales(sources, tokens, hidden, scale_dtype):
    import torch

    groups = (hidden + 127) // 128
    if scale_dtype == "int32":
        return torch.full((len(sources), groups), 0x7F7F7F7F, dtype=torch.int32)
    # Finite, positive non-unit values with distinct low mantissa bits per row/group.
    bits = (
        0x3F000001
        + torch.as_tensor(sources)[:, None] * 65536
        + torch.as_tensor(tokens)[:, None] * 128
        + torch.arange(groups)
    )
    return bits.to(torch.int32).contiguous().view(torch.float32)


def expected_combine(routes, rank, hidden, dtype, fp32_scales=False, *, weighted=False,
                     reduce_in_local=False, experts=None, bias=0):
    """Reference sum; routing weights only apply when the caller requests them.

    ``weighted=True`` models caller-side multiplication of each expert output
    by the test weight ``(slot + 1) / 32``, including its dtype conversion.
    The communication operation itself never performs that multiplication.
    Native EP first reduces experts on one destination rank and converts that
    partial sum to the wire dtype; ``reduce_in_local=True`` models this step.
    """
    import torch

    b = len(routes[rank])
    k = len(routes[rank][0]) if b else 0
    x = payload([rank] * b, list(range(b)), hidden, dtype).float()
    if fp32_scales:
        x *= fp8_scales([rank] * b, list(range(b)), hidden, "float32").repeat_interleave(128, dim=1)[:, :hidden]
    if reduce_in_local and (experts is None or experts % len(routes)):
        raise ValueError('Native local reduction requires experts divisible by world size')
    expected = torch.full_like(x, bias)
    for token in range(b):
        partials = {}
        for slot in range(k):
            expert = routes[rank][token][slot]
            if expert >= 0:
                expert_value = (x[token] * ((expert % 4 + 1) / 4) + (expert % 3 - 1) / 8).to(dtype).float()
                if weighted:
                    expert_value = (expert_value * ((slot + 1) / 32)).to(dtype).float()
                destination = expert // (experts // len(routes)) if reduce_in_local else slot
                if destination in partials:
                    expert_value = partials[destination][1] + expert_value
                partials[destination] = (slot, expert_value)
        # Native reverse routing places a rank's partial in its last valid top-k
        # slot, then the epilogue accumulates these slots in their original order.
        for _, value in sorted(partials.values(), key=lambda item: item[0]):
            expected[token] += value.to(dtype).float() if reduce_in_local else value
    return expected.to(dtype)


def prepare_experts(recv, counts, rank, experts, world, expert_dtype, *,
                    expert_alignment=1, weighted=False, routes=None):
    import torch

    values = recv[0] if isinstance(recv, tuple) else recv
    output = torch.zeros(values.shape, dtype=expert_dtype, device=values.device)
    start = 0
    for local, count in enumerate(counts):
        expert = rank * (experts // world) + local
        block = values[start : start + count].float()
        if isinstance(recv, tuple) and recv[1].dtype == torch.float32:
            block = block * recv[1][start : start + count].repeat_interleave(128, dim=1)[:, : values.shape[1]]
        expert_output = (block * ((expert % 4 + 1) / 4) + (expert % 3 - 1) / 8).to(expert_dtype)
        if weighted and count:
            if routes is None:
                raise ValueError('Caller-side weighting requires the independent route table')
            # Payload row identifiers are read before applying FP8 scales or
            # the expert transform. This also handles nondeterministic order.
            ids = values[start : start + count, :4].float().cpu().long().tolist()
            slot_weights = []
            for source_hi, source_lo, token_hi, token_lo in ids:
                source, token = source_hi * 16 + source_lo, token_hi * 16 + token_lo
                slot = routes[source][token].index(expert)
                slot_weights.append((slot + 1) / 32)
            weights = torch.tensor(slot_weights, dtype=torch.float32, device=values.device)
            expert_output = (expert_output.float() * weights[:, None]).to(expert_dtype)
        output[start : start + count] = expert_output
        start += (count + expert_alignment - 1) // expert_alignment * expert_alignment
    return output
