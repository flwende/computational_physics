#pragma once

#include <cstdint>
#include <type_traits>
#include <immintrin.h>

#if defined(__AVX512F__)
    #define SIMD_WIDTH_NATIVE_64BIT 8
    #define SIMD_WIDTH_NATIVE_32BIT 16
    #define SIMD_WIDTH_NATIVE_16BIT 32
    #define SIMD_WIDTH_NATIVE_8BIT 64
    #define SIMD_ALIGNMENT 64
#elif defined(__AVX__)
    #define SIMD_WIDTH_NATIVE_64BIT 4
    #define SIMD_WIDTH_NATIVE_32BIT 8
    #define SIMD_WIDTH_NATIVE_16BIT 16
    #define SIMD_WIDTH_NATIVE_8BIT 32
    #define SIMD_ALIGNMENT 32
#elif defined(__SSE__)
    #define SIMD_WIDTH_NATIVE_64BIT 2
    #define SIMD_WIDTH_NATIVE_32BIT 4
    #define SIMD_WIDTH_NATIVE_16BIT 8
    #define SIMD_WIDTH_NATIVE_8BIT 16
    #define SIMD_ALIGNMENT 16
#else
    #define SIMD_WIDTH_NATIVE_64BIT 1
    #define SIMD_WIDTH_NATIVE_32BIT 1
    #define SIMD_WIDTH_NATIVE_16BIT 1
    #define SIMD_WIDTH_NATIVE_8BIT 1
    #define SIMD_ALIGNMENT 8
#endif

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace simd
    {
        // Memory alignment.
        static constexpr auto alignment = std::size_t{SIMD_ALIGNMENT};

        // Data types that support SIMD operations (implementations).
        template <typename T>
        struct Implementation
        {
            static constexpr auto available = false;
        };

        #define MACRO(T)                                            \
        template <>                                                 \
        struct Implementation<T>                                    \
        {                                                           \
            static constexpr auto available = true;                 \
        };

        MACRO(double)
        MACRO(float)
        MACRO(std::uint64_t)
        MACRO(std::int64_t)
        MACRO(std::uint32_t)
        MACRO(std::int32_t)
        MACRO(std::uint16_t)
        MACRO(std::int16_t)
        MACRO(std::uint8_t)
        MACRO(std::int8_t)

        #undef MACRO

        // Get information about data types when used in the SIMD context.
        template <typename T>
        struct Type
        {
            static_assert(Implementation<T>::available, "Error: Data type not supported for SIMD.");
        };

        #define MACRO(T, WIDTH)                                     \
        template <>                                                 \
        struct Type<T>                                              \
        {                                                           \
            static constexpr auto width = std::uint32_t{WIDTH};     \
        };

        MACRO(double, SIMD_WIDTH_NATIVE_64BIT)
        MACRO(float, SIMD_WIDTH_NATIVE_32BIT)
        MACRO(std::uint64_t, SIMD_WIDTH_NATIVE_64BIT)
        MACRO(std::int64_t, SIMD_WIDTH_NATIVE_64BIT)
        MACRO(std::uint32_t, SIMD_WIDTH_NATIVE_32BIT)
        MACRO(std::int32_t, SIMD_WIDTH_NATIVE_32BIT)
        MACRO(std::uint16_t, SIMD_WIDTH_NATIVE_16BIT)
        MACRO(std::int16_t, SIMD_WIDTH_NATIVE_16BIT)
        MACRO(std::uint8_t, SIMD_WIDTH_NATIVE_8BIT)
        MACRO(std::int8_t, SIMD_WIDTH_NATIVE_8BIT)

        #undef MACRO
    }

    template <typename T>
    concept SimdVector_128Bit = requires
    {
        requires std::is_same_v<T, __m128i> || std::is_same_v<T, __m128> || std::is_same_v<T, __m128d>;
    };

    template <typename T>
    concept SimdVector_256Bit = requires
    {
        requires std::is_same_v<T, __m256i> || std::is_same_v<T, __m256> || std::is_same_v<T, __m256d>;
    };

    template <typename T>
    concept SimdVector_512Bit = requires
    {
        requires std::is_same_v<T, __m512i> || std::is_same_v<T, __m512> || std::is_same_v<T, __m512d>;
    };

    template <typename T>
    concept SimdVector = SimdVector_128Bit<T> || SimdVector_256Bit<T> || SimdVector_512Bit<T>;
}

#undef SIMD_WIDTH_NATIVE_64BIT
#undef SIMD_WIDTH_NATIVE_32BIT
#undef SIMD_WIDTH_NATIVE_16BIT
#undef SIMD_WIDTH_NATIVE_8BIT
#undef SIMD_ALIGNMENT

#undef XXX_NAMESPACE

#include "arithmetic/arithmetic.hpp"
#include "load_store/load_store.hpp"
#include "logical/logical.hpp"
#include "rotate/rotate.hpp"
