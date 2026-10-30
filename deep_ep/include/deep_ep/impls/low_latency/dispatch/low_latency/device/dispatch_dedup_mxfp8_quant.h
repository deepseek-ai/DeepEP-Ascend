// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_KERNELS_DISPATCH_DEDUP_EP_MXFP8_QUANT_H
#define DISPATCH_DEDUP_EP_KERNELS_DISPATCH_DEDUP_EP_MXFP8_QUANT_H

#include <kernel_operator.h>

namespace DispatchDedupMxfp8Quant {
using namespace AscendC;

constexpr int kPairCount = 2;
constexpr uint16_t kBf16ExponentMask = 0x7f80;
constexpr uint16_t kBf16ExponentBias = 0x7f00;
constexpr uint16_t kFp8ExponentMask = 0x00ff;
constexpr uint16_t kCustomizedNan = 0x7f81;
constexpr uint16_t kSpecialExponentThreshold = 0x0040;
constexpr int16_t kBf16ExponentShift = 7;
constexpr uint16_t kFp8E4M3MaxExponent = 0x0400;
constexpr uint16_t kFp8E5M2MaxExponent = 0x0780;
constexpr uint16_t kInvalidFp16 = 0x7c00;
constexpr int64_t kOutputElementsPerBlock = 64;

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVectorRegisterSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline void ComputeMaxExp(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUb)
{
    const uint32_t elementsPerRegister = GetVectorRegisterSize() / sizeof(T);
    const uint16_t reducedElements = GetVectorRegisterSize() / GetUbBlockSize();
    const uint16_t loopCount =
        static_cast<uint16_t>((totalCountInUb + 2 * elementsPerRegister - 1) / (2 * elementsPerRegister));

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> input0;
        MicroAPI::RegTensor<T> input1;
        MicroAPI::RegTensor<bfloat16_t> inputBf160;
        MicroAPI::RegTensor<bfloat16_t> inputBf161;
        MicroAPI::RegTensor<uint16_t> selected0;
        MicroAPI::RegTensor<uint16_t> selected1;
        MicroAPI::RegTensor<uint16_t> exponent0;
        MicroAPI::RegTensor<uint16_t> exponent1;
        MicroAPI::RegTensor<uint16_t> exponentMask;
        MicroAPI::Duplicate(exponentMask, kBf16ExponentMask);
        MicroAPI::RegTensor<uint16_t> invalidFp16;
        MicroAPI::Duplicate(invalidFp16, kInvalidFp16);
        MicroAPI::RegTensor<uint16_t> maxExponent;
        MicroAPI::MaskReg mask0;
        MicroAPI::MaskReg mask1;
        MicroAPI::MaskReg valid0;
        MicroAPI::MaskReg valid1;
        MicroAPI::UnalignReg unalign;
        static constexpr MicroAPI::CastTrait kHalfToBf16 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t loop = 0; loop < loopCount; ++loop) {
            mask0 = MicroAPI::UpdateMask<T>(totalCountInUb);
            mask1 = MicroAPI::UpdateMask<T>(totalCountInUb);
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                input0, input1, srcAddr, elementsPerRegister * kPairCount);
            if constexpr (Std::IsSame<T, half>::value) {
                MicroAPI::And(selected0, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(input0), invalidFp16, mask0);
                MicroAPI::And(selected1, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(input1), invalidFp16, mask0);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(valid0, selected0, invalidFp16, mask0);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(valid1, selected1, invalidFp16, mask0);
                MicroAPI::Cast<bfloat16_t, T, kHalfToBf16>(inputBf160, input0, mask0);
                MicroAPI::Cast<bfloat16_t, T, kHalfToBf16>(inputBf161, input1, mask0);
                MicroAPI::And(exponent0, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(inputBf160), exponentMask,
                              mask0);
                MicroAPI::And(exponent1, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(inputBf161), exponentMask,
                              mask0);
                MicroAPI::Select<uint16_t>(exponent0, exponent0, exponentMask, valid0);
                MicroAPI::Select<uint16_t>(exponent1, exponent1, exponentMask, valid1);
            } else {
                MicroAPI::And(exponent0, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(input0), exponentMask, mask0);
                MicroAPI::And(exponent1, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(input1), exponentMask, mask0);
            }
            MicroAPI::Max(maxExponent, exponent0, exponent1, mask0);
            MicroAPI::ReduceMaxWithDataBlock(maxExponent, maxExponent, mask0);
            MicroAPI::DataCopyUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, maxExponent,
                                                                                         unalign, reducedElements);
        }
        MicroAPI::DataCopyUnAlignPost(maxExpAddr, unalign, 0);
    }
}

template <typename Fp8Type>
__aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUb)
{
    const uint32_t elementsPerRegister = GetVectorRegisterSize() / sizeof(uint16_t);
    const uint16_t loopCount = static_cast<uint16_t>((totalScaleInUb + elementsPerRegister - 1) / elementsPerRegister);
    const uint16_t maxFp8Exponent =
        Std::IsSame<Fp8Type, fp8_e4m3fn_t>::value ? kFp8E4M3MaxExponent : kFp8E5M2MaxExponent;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> exponentMask;
        MicroAPI::RegTensor<uint16_t> maxExponent;
        MicroAPI::Duplicate(exponentMask, kBf16ExponentMask);
        MicroAPI::MaskReg validMask;
        MicroAPI::MaskReg nonzeroMask;
        MicroAPI::MaskReg boundedMask;
        MicroAPI::MaskReg specialMask;
        MicroAPI::RegTensor<uint16_t> maxFp8ExponentTensor;
        MicroAPI::Duplicate(maxFp8ExponentTensor, maxFp8Exponent);
        MicroAPI::RegTensor<uint16_t> sharedExponent;
        MicroAPI::RegTensor<uint16_t> scaleValue;
        MicroAPI::RegTensor<uint16_t> exponentBias;
        MicroAPI::Duplicate(exponentBias, kBf16ExponentBias);
        MicroAPI::RegTensor<uint16_t> reciprocalScale;
        MicroAPI::RegTensor<uint16_t> fp8Nan;
        MicroAPI::Duplicate(fp8Nan, kFp8ExponentMask);
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::Duplicate(zero, 0);
        MicroAPI::RegTensor<uint16_t> nan;
        MicroAPI::Duplicate(nan, kCustomizedNan);
        MicroAPI::RegTensor<uint16_t> specialExponent;
        MicroAPI::Duplicate(specialExponent, kSpecialExponentThreshold);

        for (uint16_t loop = 0; loop < loopCount; ++loop) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<uint16_t>(totalScaleInUb);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExponent, maxExpAddr,
                                                                                  elementsPerRegister);
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(validMask, maxExponent, exponentMask, mask);
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(nonzeroMask, maxExponent, zero, mask);
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(boundedMask, maxExponent, maxFp8ExponentTensor, mask);
            MicroAPI::Select<uint16_t>(maxExponent, maxFp8ExponentTensor, maxExponent, boundedMask);
            MicroAPI::Sub(sharedExponent, maxExponent, maxFp8ExponentTensor, mask);
            MicroAPI::ShiftRights(scaleValue, sharedExponent, kBf16ExponentShift, mask);
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8Nan, validMask);
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zero, nonzeroMask);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, elementsPerRegister / kPairCount, mask);

            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExponent, exponentBias, mask);
            MicroAPI::Sub(reciprocalScale, exponentBias, sharedExponent, mask);
            MicroAPI::Select<uint16_t>(reciprocalScale, reciprocalScale, nan, validMask);
            MicroAPI::Select<uint16_t>(reciprocalScale, reciprocalScale, zero, nonzeroMask);
            MicroAPI::Select<uint16_t>(reciprocalScale, specialExponent, reciprocalScale, specialMask);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, reciprocalScale,
                                                                                  elementsPerRegister, mask);
        }
    }
}

template <typename InputType, typename Fp8Type, RoundMode ToBf16RoundMode, RoundMode Fp8RoundMode>
__aicore__ inline void ComputeFp8Data(__ubuf__ InputType* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                      __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUb)
{
    const uint32_t elementsPerRegister = GetVectorRegisterSize() / sizeof(InputType);
    const uint16_t scaleElementsPerBlock = GetVectorRegisterSize() / GetUbBlockSize();
    uint32_t doubledCount = totalCountInUb * kPairCount;
    const uint16_t loopCount =
        static_cast<uint16_t>((totalCountInUb + 2 * elementsPerRegister - 1) / (2 * elementsPerRegister));

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg inputMask0;
        MicroAPI::MaskReg inputMask1;
        MicroAPI::MaskReg fp32Mask0;
        MicroAPI::MaskReg fp32Mask1;
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint16_t> packedScale;
        MicroAPI::RegTensor<float> fp32Scale;
        MicroAPI::RegTensor<InputType> input0;
        MicroAPI::RegTensor<InputType> input1;
        MicroAPI::RegTensor<float> fp32Input00;
        MicroAPI::RegTensor<float> fp32Input01;
        MicroAPI::RegTensor<float> fp32Input10;
        MicroAPI::RegTensor<float> fp32Input11;
        MicroAPI::RegTensor<Fp8Type> fp8Input00;
        MicroAPI::RegTensor<Fp8Type> fp8Input01;
        MicroAPI::RegTensor<Fp8Type> fp8Input10;
        MicroAPI::RegTensor<Fp8Type> fp8Input11;
        static constexpr MicroAPI::CastTrait kEvenElements = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait kOddElements = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait kFp32ToFp8 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                           MicroAPI::MaskMergeMode::ZEROING, Fp8RoundMode};

        for (uint16_t loop = 0; loop < loopCount; ++loop) {
            inputMask0 = MicroAPI::UpdateMask<InputType>(totalCountInUb);
            inputMask1 = MicroAPI::UpdateMask<InputType>(totalCountInUb);
            fp32Mask0 = MicroAPI::UpdateMask<InputType>(doubledCount);
            fp32Mask1 = MicroAPI::UpdateMask<InputType>(doubledCount);
            MicroAPI::DataCopy<InputType, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                input0, input1, srcAddr, elementsPerRegister * kPairCount);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                packedScale, halfScaleLocalAddr, scaleElementsPerBlock);

            if constexpr (Std::IsSame<InputType, half>::value) {
                MicroAPI::Cast<float, InputType, kEvenElements>(fp32Input00, input0, inputMask0);
                MicroAPI::Cast<float, InputType, kOddElements>(fp32Input01, input0, inputMask0);
                MicroAPI::Cast<float, bfloat16_t, kEvenElements>(
                    fp32Scale, reinterpret_cast<MicroAPI::RegTensor<bfloat16_t>&>(packedScale), allMask);
                MicroAPI::Mul(fp32Input00, fp32Input00, fp32Scale, fp32Mask0);
                MicroAPI::Mul(fp32Input01, fp32Input01, fp32Scale, fp32Mask1);
                MicroAPI::Interleave(fp32Input00, fp32Input01, fp32Input00, fp32Input01);
                MicroAPI::Cast<float, InputType, kEvenElements>(fp32Input10, input1, inputMask0);
                MicroAPI::Cast<float, InputType, kOddElements>(fp32Input11, input1, inputMask0);
                MicroAPI::Mul(fp32Input10, fp32Input10, fp32Scale, fp32Mask0);
                MicroAPI::Mul(fp32Input11, fp32Input11, fp32Scale, fp32Mask1);
                MicroAPI::Interleave(fp32Input10, fp32Input11, fp32Input10, fp32Input11);
                MicroAPI::Interleave(fp32Input00, fp32Input10, fp32Input00, fp32Input10);
                MicroAPI::Interleave(fp32Input01, fp32Input11, fp32Input01, fp32Input11);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input00, fp32Input00, fp32Mask0);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input01, fp32Input10, fp32Mask0);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input10, fp32Input01, fp32Mask1);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input11, fp32Input11, fp32Mask1);
            } else {
                MicroAPI::Mul(input0, input0, reinterpret_cast<MicroAPI::RegTensor<InputType>&>(packedScale),
                              inputMask0);
                MicroAPI::Mul(input1, input1, reinterpret_cast<MicroAPI::RegTensor<InputType>&>(packedScale),
                              inputMask0);
                MicroAPI::Interleave(input0, input1, input0, input1);
                MicroAPI::Cast<float, InputType, kEvenElements>(fp32Input00, input0, inputMask0);
                MicroAPI::Cast<float, InputType, kOddElements>(fp32Input01, input0, inputMask0);
                MicroAPI::Interleave(fp32Input00, fp32Input01, fp32Input00, fp32Input01);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input00, fp32Input00, fp32Mask0);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input01, fp32Input01, fp32Mask0);
                MicroAPI::Cast<float, InputType, kEvenElements>(fp32Input10, input1, inputMask1);
                MicroAPI::Cast<float, InputType, kOddElements>(fp32Input11, input1, inputMask1);
                MicroAPI::Interleave(fp32Input10, fp32Input11, fp32Input10, fp32Input11);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input10, fp32Input10, fp32Mask1);
                MicroAPI::Cast<Fp8Type, float, kFp32ToFp8>(fp8Input11, fp32Input11, fp32Mask1);
            }

            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, reinterpret_cast<MicroAPI::RegTensor<int8_t>&>(fp8Input00), kOutputElementsPerBlock,
                fp32Mask0);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, reinterpret_cast<MicroAPI::RegTensor<int8_t>&>(fp8Input01), kOutputElementsPerBlock,
                fp32Mask0);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, reinterpret_cast<MicroAPI::RegTensor<int8_t>&>(fp8Input10), kOutputElementsPerBlock,
                fp32Mask1);
            MicroAPI::DataCopy<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, reinterpret_cast<MicroAPI::RegTensor<int8_t>&>(fp8Input11), kOutputElementsPerBlock,
                fp32Mask1);
        }
    }
}

template <typename Fp8Type, bool ApplyExpertScale = true>
__aicore__ inline void DequantizeAndAccumulate(__ubuf__ uint8_t* tokenAddr, __ubuf__ fp8_e8m0_t* scaleAddr,
                                               __ubuf__ float* scaleWorkAddr, __ubuf__ float* sumAddr, uint32_t axisH,
                                               uint32_t scaleCount, float expertScale)
{
    (void)expertScale;
    const uint32_t fp32RepeatSize = GetVectorRegisterSize() / sizeof(float);
    const uint16_t scaleRepeatTimes = static_cast<uint16_t>((scaleCount + fp32RepeatSize - 1U) / fp32RepeatSize);
    const uint16_t dataRepeatTimes = static_cast<uint16_t>((axisH + fp32RepeatSize * 2U - 1U) / (fp32RepeatSize * 2U));
    uint32_t scaleMaskCount = scaleCount;
    uint32_t dataMaskCount = axisH;
    uint32_t fp8MaskCount = axisH * 4U;
    constexpr int16_t kFp32ExponentShift = 23;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<fp8_e8m0_t> scaleReg;
        MicroAPI::RegTensor<Fp8Type> tokenReg;
        MicroAPI::RegTensor<float> tokenFp320;
        MicroAPI::RegTensor<float> tokenFp321;
        MicroAPI::RegTensor<float> scaleFp32;
        MicroAPI::RegTensor<float> weighted0;
        MicroAPI::RegTensor<float> weighted1;
        MicroAPI::RegTensor<float> sum0;
        MicroAPI::RegTensor<float> sum1;
        static constexpr MicroAPI::CastTrait kCastLane0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr MicroAPI::CastTrait kCastLane2 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t loop = 0; loop < scaleRepeatTimes; ++loop) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(scaleMaskCount);
            MicroAPI::DataCopy<fp8_e8m0_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(scaleReg,
                                                                                scaleAddr + loop * fp32RepeatSize);
            MicroAPI::ShiftLefts(reinterpret_cast<MicroAPI::RegTensor<uint32_t>&>(scaleFp32),
                                 reinterpret_cast<MicroAPI::RegTensor<uint32_t>&>(scaleReg), kFp32ExponentShift, mask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(scaleWorkAddr + loop * fp32RepeatSize * 2U,
                                                                           scaleFp32, scaleFp32, mask);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t loop = 0; loop < dataRepeatTimes; ++loop) {
            MicroAPI::MaskReg dataMask = MicroAPI::UpdateMask<float>(dataMaskCount);
            MicroAPI::MaskReg fp8Mask = MicroAPI::UpdateMask<Fp8Type>(fp8MaskCount);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(scaleFp32, scaleWorkAddr + loop * 8U);
            MicroAPI::DataCopy<Fp8Type, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                tokenReg, reinterpret_cast<__ubuf__ Fp8Type*>(tokenAddr) + 2U * loop * fp32RepeatSize);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(sum0, sum1,
                                                                           sumAddr + 2U * loop * fp32RepeatSize);
            MicroAPI::Cast<float, Fp8Type, kCastLane0>(tokenFp320, tokenReg, fp8Mask);
            MicroAPI::Cast<float, Fp8Type, kCastLane2>(tokenFp321, tokenReg, fp8Mask);
            MicroAPI::Mul(weighted0, scaleFp32, tokenFp320, dataMask);
            MicroAPI::Mul(weighted1, scaleFp32, tokenFp321, dataMask);
            if constexpr (ApplyExpertScale) {
                MicroAPI::Muls(weighted0, weighted0, expertScale, dataMask);
                MicroAPI::Muls(weighted1, weighted1, expertScale, dataMask);
            }
            MicroAPI::Add(sum0, sum0, weighted0, dataMask);
            MicroAPI::Add(sum1, sum1, weighted1, dataMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(sumAddr + loop * fp32RepeatSize * 2U, sum0,
                                                                           sum1, dataMask);
        }
    }
}

}  // namespace DispatchDedupMxfp8Quant

#endif  // DISPATCH_DEDUP_EP_KERNELS_DISPATCH_DEDUP_EP_MXFP8_QUANT_H
