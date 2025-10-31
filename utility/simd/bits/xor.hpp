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
        inline SimdVector auto VecXor(const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t 
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_xor_epi32(vec_a, vec_b);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_xor_si256(vec_a, vec_b);
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
