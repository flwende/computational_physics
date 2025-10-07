#pragma once

#include <array>

#include "set_1.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Set vector values.
        template <typename ElementType, std::size_t N>
        inline SimdVector auto VecSet(const std::array<ElementType, N>& values)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            static_assert(N == VecWidth, "Provided and expected number of elements does not match.");

            if constexpr (std::is_integral_v<ElementType>)
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_loadu_si512(static_cast<const void*>(values.data()));
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data()));
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
