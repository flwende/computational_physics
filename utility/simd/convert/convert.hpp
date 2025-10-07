#pragma once

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Convert from SrcElementType to DstElementType.
        template <typename SrcElementType, typename DstElementType>
        inline SimdVector auto VecConvert(const SimdVector auto& vec)
        {
            constexpr auto VecWidth = simd::Type<DstElementType>::Width;
            constexpr auto VecBits = VecWidth * 8 * sizeof(DstElementType);

            if constexpr (std::is_integral_v<DstElementType> && sizeof(DstElementType) == 4) // std::int32_t, std::uint32_t
            {
                if constexpr (std::is_integral_v<SrcElementType> && sizeof(SrcElementType) == 1) // std::int8_t, std::uint8_t
                {
                    if constexpr (VecBits == 512)
                    {
                        return _mm512_cvtepi8_epi32(vec);
                    }
                    else if constexpr (VecBits == 256)
                    {
                        return _mm256_cvtepi8_epi32(vec);
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
            else
            {
                static_assert(false, "Not implemented.");
            }
        }
    }
}

#undef XXX_NAMESPACE
