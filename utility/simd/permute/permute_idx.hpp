#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Permutation of vector elements.
        template <typename ElementType>
        inline SimdVector auto VecPermuteIdx(const SimdVector auto& vec, const SimdVector auto& idx)
        {
            using IdxType = std::remove_cvref_t<decltype(idx)>;

            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512 && std::is_same_v<IdxType, __m512i>)
                {
                    return _mm512_permutexvar_epi32(idx, vec);
                }
                else if constexpr (VecBits == 256 && std::is_same_v<IdxType, __m256i>)
                {
                    return _mm256_permutevar8x32_epi32(vec, idx);
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

#include "rotate_right.hpp"
