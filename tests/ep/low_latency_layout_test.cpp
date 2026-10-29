// Host-only layout checks; no Torch, CANN or NPU required.
// c++ -std=c++17 -Ideep_ep/include tests/ep/low_latency_layout_test.cpp -o /tmp/ll-layout
#include <cassert>
#include <iostream>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/comm_args.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h>
#include <deep_ep/impls/low_latency/common/ep_server_order.h>
#include <deep_ep/impls/low_latency/common/ep_packet_receive_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_layout.h>
#include <deep_ep/impls/low_latency/combine/low_latency/combine_layout.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_tiling.h>

namespace {
void check_upstream_protocol() {
    using namespace DispatchDedup;
    assert(!EpServerDedupEnabled(1) && !EpServerDedupEnabled(32));
    assert(EpServerDedupEnabled(33) && EpServerDedupEnabled(256));
    for (uint32_t experts: {8U, 16U, 128U, 256U, 1024U}) {
        const auto counts = DispatchServerDedupCountLayout(experts);
        assert(counts.cntOffset % 512U == 0U);
        // No two source count records share a 512-byte GM slot.
        assert(kEpServerDedupCountGmStride == 512U);
        assert(counts.cumsumOffset >= uint64_t(experts) * 512U);
        assert(counts.stateBytes <= COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET);
    }
    using PacketReceiveLayout::State;
    for (uint32_t blocks: {1U, 2U, 16U, 31U, 64U}) {
        assert(PacketReceiveLayout::Decode(0, blocks) == State::Pending);
        assert(PacketReceiveLayout::Decode(float(blocks * 8U - 1), blocks) == State::Pending);
        assert(PacketReceiveLayout::Decode(float(blocks * 8U), blocks) == State::Compact);
        assert(PacketReceiveLayout::Decode(float(blocks * 8U) + 0.5f, blocks) == State::KeepFlags);
    }
    // EP8 has one server; include odd counts to exercise the shared helper's
    // unmatched half, and EP256's 32 servers without 32-bit mask overflow.
    for (uint32_t servers = 1; servers <= 32; ++servers)
        for (uint32_t own = 0; own < servers; ++own) {
            uint64_t dedup = 0, direct = 0;
            for (uint32_t step = 0; step < servers; ++step) {
                const auto a = EpServerOrder::ServerDedupHalfOrder(servers, own, step);
                const auto b = EpServerOrder::ServerDedupDispatchDirectOrder(servers, own, step);
                assert(a < servers && b < servers);
                assert(!(dedup & (1ULL << a)) && !(direct & (1ULL << b)));
                dedup |= 1ULL << a;
                direct |= 1ULL << b;
                if (servers < 16) assert(b == (own + step) % servers);
                else if (servers % 2 == 0)
                    assert((a < servers / 2) == ((step % 2 == 0) == (own < servers / 2)));
            }
            assert(dedup == (1ULL << servers) - 1 && direct == dedup);
        }
}

bool overlaps(uint32_t a, uint32_t a_bytes, uint32_t b, uint32_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
}

void check_receive_workspace(const CombineDedup::EpCombineM45WorkspaceLayout& shared, uint32_t budget) {
    using namespace CombineDedup;
    assert(shared.arenaBytes == budget);
    assert(shared.inputLanes >= 1 && shared.inputLanes <= kEpServerDedupCombineM4MaxInputLanes);
    assert(shared.outputLanes >= 1 && shared.outputLanes <= 8);
    assert(shared.totalBytes <= budget);
    assert(shared.inputOffset >= shared.phaseOffset + shared.phaseBytes);
    assert(shared.outputOffset >= shared.inputOffset + shared.inputLanes * shared.inputBytes);
    assert(kEpServerDedupCombineResidentUbBytes + shared.outputOffset +
               shared.outputLanes * shared.outputBytes <= budget);
    // Cover every output-ring position, including fewer tasks than slots.
    for (uint32_t tasks = 0; tasks <= 2 * shared.outputLanes + 1; ++tasks) {
        const auto final = EpCombineM5Layout(shared, tasks);
        assert(final.inputLanes >= 1 && final.inputLanes <= kEpServerDedupCombineM5MaxInputLanes);
        for (uint32_t lane = 0; lane < final.inputLanes; ++lane) {
            const auto offset = final.inputOffsets[lane];
            assert(offset >= shared.finalInputOffset);
            assert(kEpServerDedupCombineResidentUbBytes + offset + shared.inputBytes <= budget);
            for (uint32_t previous = 0; previous < lane; ++previous)
                assert(!overlaps(offset, shared.inputBytes, final.inputOffsets[previous], shared.inputBytes));
            uint32_t retire_mask = 0;
            for (uint32_t slot = 0; slot < shared.outputLanes; ++slot)
                if (overlaps(offset, shared.inputBytes, shared.outputOffset + slot * shared.outputBytes,
                             shared.outputBytes))
                    retire_mask |= 1U << slot;
            assert(final.inputRetireMasks[lane] == retire_mask);
            for (uint32_t output = 0; output < shared.finalOutputLanes; ++output) {
                assert(final.outputSlots[output] < shared.outputLanes);
                assert(!(retire_mask & (1U << final.outputSlots[output])));
            }
        }
    }
}

void check_ub_budgets() {
    using namespace CombineDedup;
    for (uint32_t budget: {190U * 1024, 192U * 1024, 216U * 1024, 240U * 1024}) {
        Layout plan{};
        // The reported EP8 smoke-test shape must work with a reduced arena.
        assert(BuildLayout(8, 16, 7168, 8, 128, plan, budget));
        check_receive_workspace(EpCombineM45Layout(1024, 16, 7168, 8, budget), budget);
        for (uint32_t tokens: {0U, 1U, 1024U, 8192U, 10000U})
            for (uint32_t bs: {1U, 16U, 32U, 255U})
                for (uint32_t h: {128U, 7168U, 12288U, 16384U, 20480U})
                    for (uint32_t k: {1U, 8U, 32U}) {
                        const auto shared = EpCombineM45Layout(tokens, bs, h, k, budget);
                        if (shared.inputLanes) check_receive_workspace(shared, budget);
                    }
    }
    for (uint32_t budget: {0U, 190U * 1024 - 32, 216U * 1024 - 1, 240U * 1024 + 32}) {
        Layout plan{};
        assert(!BuildLayout(8, 16, 7168, 8, 128, plan, budget));
        assert(EpCombineM45Layout(1024, 16, 7168, 8, budget).inputLanes == 0);
    }
    const auto reduced = EpCombineM45Layout(1024, 16, 7168, 8, 216U * 1024);
    const auto original = EpCombineM45Layout(1024, 16, 7168, 8, 240U * 1024);
    check_receive_workspace(reduced, 216U * 1024);
    check_receive_workspace(original, 240U * 1024);
    // Removing weight multiplication eliminates its FP32 temporary. H12288
    // now fits a 190 KiB arena; validate offsets and lanes, not the old 3H budget.
    for (uint32_t budget: {190U * 1024, 240U * 1024}) {
        const auto wide = EpCombineM45Layout(1024, 16, 12288, 8, budget);
        assert(wide.inputLanes > 0 && wide.outputLanes > 0);
        check_receive_workspace(wide, budget);
    }
    // A BF16 row alone exceeds the entire arena at this size. It must still
    // be rejected, regardless of how the remaining temporaries are arranged.
    assert(EpCombineM45Layout(1024, 16, 240U * 1024, 8, 240U * 1024).inputLanes == 0);
}
}  // namespace

int main() {
    check_upstream_protocol();
    check_ub_budgets();
    using namespace DispatchDedup;
    for (int ranks: {8, 16, 32, 64, 128, 256}) {
        const int experts = std::max(128, ranks);
        for (const auto mode: {EpServerDedupInputMode::Plain16, EpServerDedupInputMode::PrequantizedFp8Packs}) {
            EpMemoryServerDedupLayout dispatch{};
            CombineDedup::Layout combine{};
            assert(BuildDispatchServerDedupLayout(ranks, 16, 7168, 8, experts, &dispatch, mode) == 0);
            assert(DispatchDedupPacketUbFits(dispatch, 16, 8, experts, ranks));
            assert(CombineDedup::BuildLayout(ranks, 16, 7168, 8, experts, combine));
            assert(dispatch.capacityRows == combine.capacityRows);
            assert(dispatch.capacityRows == uint64_t(ranks * 16 * std::min(8, experts / ranks)));
            assert(combine.requiredWindowBytes <= kEpServerDedupWindowUsableBytes);
        }
    }
    for (int ranks: {0, 1, 2, 4, 7, 9, 129, 257}) {
        EpMemoryServerDedupLayout dispatch{};
        CombineDedup::Layout combine{};
        assert(BuildDispatchServerDedupLayout(ranks, 16, 7168, 8, 128, &dispatch, false) != 0);
        assert(!CombineDedup::BuildLayout(ranks, 16, 7168, 8, 128, combine));
    }
    EpMemoryServerDedupLayout plan{};
    CombineDedup::Layout full_batch{};
    assert(BuildDispatchServerDedupLayout(8, 256, 7168, 8, 8, &plan, false) == 0);
    assert(DispatchDedupPacketUbFits(plan, 256, 8, 8, 8));
    assert(CombineDedup::BuildLayout(8, 256, 7168, 8, 8, full_batch, 190U * 1024));
    assert(BuildDispatchServerDedupLayout(8, 0, 7168, 8, 128, &plan, false) != 0);
    assert(BuildDispatchServerDedupLayout(8, 16, 7168, 8, 127, &plan, false) != 0);
    assert(BuildDispatchServerDedupLayout(8, 16, 7168, 33, 128, &plan, false) != 0);
    assert(BuildDispatchServerDedupLayout(8, 16, 1000000, 8, 128, &plan, false) != 0 ||
           !DispatchDedupPacketUbFits(plan, 16, 8, 128, 8));
    assert(DispatchServerDedupWindowOffset(0) == 0);
    assert(DispatchServerDedupWindowOffset(1) == (4ULL << 30));
    assert(DispatchServerDedupWindowOffset(2) == 0);
    std::cout << "LL EP8..256 layouts, upstream protocol and 190/192/216/240 KiB UB budgets passed\n";
}
