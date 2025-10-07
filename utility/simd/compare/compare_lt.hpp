#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Compare two vectors: Lower Than.
        template <typename ElementType>
        inline SimdMask auto VecCompareLT(const SimdVector auto& vec_a, const SimdVector auto& vec_b)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_same_v<ElementType, float>) // float
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_cmp_ps_mask(vec_a, vec_b, _CMP_LT_OS);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_castps_si256(_mm256_cmp_ps(vec_a, vec_b, _CMP_LT_OS));
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
