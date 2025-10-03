#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Element-wise shift to the right by 'value' with wrap-around semantics: rotate.
        template <typename ElementType>
        inline SimdVector auto VecRotateRight(const SimdVector auto& vec, const std::uint32_t value)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType>)
            {
                alignas(simd::alignment) std::uint32_t idx[VecWidth];
                for (std::uint32_t i = 0; i < VecWidth; ++i)
                    idx[i] = i;

                if constexpr (VecBits == 512)
                {
                    auto v_idx = _mm512_load_epi32((const void*)&idx[0]);
                    v_idx = _mm512_and_epi32(_mm512_add_epi32(v_idx, _mm512_set1_epi32(value)), _mm512_set1_epi32(VecWidth - 1));
                    return _mm512_permutexvar_epi32(v_idx, vec);
                }
                else if constexpr (VecBits == 256)
                {
                    auto v_idx = _mm256_load_si256((const __m256i*)&idx[0]);
                    v_idx = _mm256_and_epi32(_mm256_add_epi32(v_idx, _mm256_set1_epi32(value)), _mm256_set1_epi32(VecWidth - 1));
                    return _mm256_permutevar8x32_epi32(vec, v_idx);
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
