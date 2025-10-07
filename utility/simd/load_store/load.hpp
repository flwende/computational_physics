#pragma once

#include <array>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Load unaligned from address pointed to by 'ptr'.
        template <typename ElementType, std::uint32_t Bits = 0>
        inline SimdVector auto VecLoad(const ElementType* ptr)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType>)
            {
                if constexpr (Bits == 64)
                {
                    std::uint64_t data[2] = {};
                    data[0] = *(reinterpret_cast<const std::uint64_t*>(ptr));
                    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(&data[0]));
                }
                else if constexpr (Bits == 128)
                {
                    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
                }
                else if constexpr (VecBits == 512)
                {
                    return _mm512_loadu_si512(static_cast<const void*>(ptr));
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
                }
                else
                {
                    static_assert(false, "Not implemented.");
                }
            }
            else if constexpr (std::is_same_v<ElementType, float>)
            {
                if constexpr (VecBits == 512)
                {
                    return _mm512_loadu_ps(static_cast<const void*>(ptr));
                }
                else if constexpr (VecBits == 256)
                {
                    return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
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

        template <typename ElementType, std::uint32_t Bits = 0, std::size_t N>
        inline SimdVector auto VecLoad(const std::array<ElementType, N>& data)
        {
            return VecLoad<ElementType, Bits>(&data[0]);
        }
    }
}

#undef XXX_NAMESPACE
