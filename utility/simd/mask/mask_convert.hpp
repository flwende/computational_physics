#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Extract integer from mask.
        template <typename ElementType>
        inline auto MaskToInteger(const SimdMask auto& mask)
        {
            using MaskType = std::remove_cvref_t<decltype(mask)>;

            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    static_assert(std::is_same_v<MaskType, __mmask16>, "Mask type __mmask16 expected.");
                    return _cvtmask16_u32(mask);
                }
                else if constexpr (VecBits == 256)
                {
                    static_assert(std::is_same_v<MaskType, __m256i>, "Mask type __m256i expected.");
                    return _mm256_movemask_ps(_mm256_castsi256_ps(mask));
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

        // Create mask from integer.
        template <typename ElementType, typename Integer>
        inline SimdMask auto MaskFromInteger(const Integer i_mask)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    return _cvtu32_mask16(static_cast<std::uint16_t>(i_mask & 0xFFFF));
                }
                else if constexpr (VecBits == 256)
                {
                    auto mask = _mm256_and_si256(_mm256_set1_epi32(i_mask), _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128));
                    return _mm256_cmpgt_epi32(mask, _mm256_setzero_si256());
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
