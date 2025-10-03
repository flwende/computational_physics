#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Store unaligned to address pointed to by 'ptr'.
        template <typename ElementType>
        inline void VecStore(ElementType* ptr, const SimdVector auto& vec)
        {
            constexpr auto VecWidth = simd::Type<ElementType>::width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(ElementType);

            if constexpr (std::is_integral_v<ElementType>)
            {
                if constexpr (VecBits == 512)
                {
                    _mm512_storeu_si512(static_cast<void*>(ptr), vec);
                }
                else if constexpr (VecBits == 256)
                {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), vec);
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
