#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        namespace
        {
            [[ maybe_unused ]] inline auto _mm256_and_epi32(const __m256i& vec_a, const __m256i& vec_b)
            {
                return _mm256_castps_si256(_mm256_and_ps(
                    _mm256_castsi256_ps(vec_a),
                    _mm256_castsi256_ps(vec_b)));
            }
        }

        // Bitwise logical AND of two vectors.
        template <typename ElementType>
        inline SimdVector auto VecAnd(const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t 
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_and_epi32(vec_a, vec_b);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_and_epi32(vec_a, vec_b);
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
