#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Compare two vectors using a mask: Not Equal.
        template <typename ElementType>
        inline SimdMask auto VecCompareNE(const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_cmp_epi32_mask(vec_a, vec_b, _MM_CMPINT_NE);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_xor_si256(_mm256_cmpeq_epi32(vec_a, vec_b), _mm256_set1_epi32(0xFFFFFFFF));
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

        // Compare two vectors using a mask: Not Equal.
        template <typename ElementType>
        inline SimdMask auto VecCompareNE_Masked(const SimdMask auto& mask, const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            using MaskType = std::remove_cvref_t<decltype(mask)>;

            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    static_assert(std::is_same_v<MaskType, __mmask16>, "Mask type __mmask16 expected.");
                    return _mm512_mask_cmp_epi32_mask(mask, vec_a, vec_b, _MM_CMPINT_NE);
                }
                else if constexpr (VecBits == 256)
                {
                    static_assert(std::is_same_v<MaskType, __m256i>, "Mask type __m256i expected.");
                    return _mm256_andnot_si256(_mm256_cmpeq_epi32(vec_a, vec_b), mask);
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
