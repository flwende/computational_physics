#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Set vector values.
        template <typename ElementType>
        inline SimdVector auto VecSet1(const ElementType value)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_set1_epi32(value);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_set1_epi32(value);
                }
                else
                {
                    static_assert(false, "Not implemented.");
                }
            }
            else if constexpr (std::is_same_v<ElementType, float>) // float
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_set1_ps(value);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_set1_ps(value);
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

        // Set vector values using a mask: inactive vector lanes are assigned zero.
        template <typename ElementType>
        inline SimdVector auto VecSet1_Masked(const SimdMask auto& mask, const ElementType value)
        {
            using MaskType = std::remove_cvref_t<decltype(mask)>;

            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    static_assert(std::is_same_v<MaskType, __mmask16>, "Mask type __mmask16 expected.");
                    return _mm512_mask_set1_epi32(_mm512_setzero_si512(), mask, value);
                }
                else if constexpr (VecBits == 256)
                {
                    static_assert(std::is_same_v<MaskType, __m256i>, "Mask type __m256i expected.");
                    return _mm256_blendv_epi8(_mm256_setzero_si256(), _mm256_set1_epi32(value), mask);
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
