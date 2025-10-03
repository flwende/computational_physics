#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Load unaligned from address pointed to by 'ptr'.
        template <typename ElementType>
        inline SimdVector auto VecLoad(const ElementType* ptr)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType>)
            {
                if constexpr (VecBits == 512)
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
            else
            {
                static_assert(false, "Not implemented.");
            }
        }
    }
}

#undef XXX_NAMESPACE
