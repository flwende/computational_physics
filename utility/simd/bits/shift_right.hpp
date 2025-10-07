#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Shift bits by Value.
        template <typename ElementType, std::uint32_t Value>
        inline SimdVector auto VecShiftRight(const SimdVector auto& vec)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType> && sizeof(ElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_srli_epi32(vec, Value);
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_srli_epi32(vec, Value);
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
