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
    }
}

#undef XXX_NAMESPACE
