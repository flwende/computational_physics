#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Apply bit-wise AND to two masks.
        template <typename ElementType>
        inline SimdMask auto MaskAnd(const SimdMask auto& mask_a, const SimdMask auto& mask_b)
        {
            using MaskType_a = std::remove_cvref_t<decltype(mask_a)>;
            using MaskType_b = std::remove_cvref_t<decltype(mask_b)>;

            static_assert(std::is_same_v<MaskType_a, MaskType_b>, "Different mask types are not allowed.");

            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    static_assert(std::is_same_v<MaskType_a, __mmask16>, "Mask type __mmask16 expected.");
                    return _kand_mask16(mask_a, mask_b);
                }
                else if constexpr (VecBits == 256)
                {
                    static_assert(std::is_same_v<MaskType_a, __m256i>, "Mask type __m256i expected.");
                    return _mm256_and_si256(mask_a, mask_b);
                }
                else
                {
                    static_assert(false, "Not implemented.");
                }
            }
            else
            {
                static_assert(false, "Not implemented.");
            }
        }
    }
}

#undef XXX_NAMESPACE
