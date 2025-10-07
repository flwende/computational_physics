#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Bitwise OR of two vectors.
        template <typename ElementType>
        inline SimdVector auto VecOr(const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t 
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_or_epi32(vec_a, vec_b);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_or_si256(vec_a, vec_b);
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

        // Bitwise OR of two vectors using a mask argument.
        template <typename ElementType>
        inline SimdVector auto VecOr_Masked(const SimdVector auto& vec, const SimdMask auto& mask, const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            using MaskType = std::remove_cvref_t<decltype(mask)>;

            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t 
            {
                if constexpr (VecBits == 512)
                {
                    static_assert(std::is_same_v<MaskType, __mmask16>, "Mask type __mmask16 expected.");
                    return _mm512_mask_or_epi32(vec, mask, vec_a, vec_b);
                }
                else if constexpr (VecBits == 256)
                {
                    static_assert(std::is_same_v<MaskType, __m256i>, "Mask type __m256i expected.");
                    return _mm256_blendv_epi8(vec, _mm256_or_si256(vec_a, vec_b), mask);
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
