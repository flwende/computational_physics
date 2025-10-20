#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    // Generate a sequence of N numbers starting from 'start' with a maximum value of 'max_value'
    // in the range std::int32_t.MIN to std::int32_t.MAX.
    template <typename OutType, std::int32_t N>
    constexpr auto Iota(const std::int32_t start, const std::int32_t max_value = std::numeric_limits<std::int32_t>::max())
        -> std::array<OutType, N>
    {
        static_assert(std::is_integral_v<OutType>, "Only integral types are supported.");
        static_assert(N >= 0, "Negative N is not allowed.");

        assert(N < (std::numeric_limits<std::int32_t>::max() - (start > 0 ? start : 0)) && "Overflow in Iota.");

        auto tmp = std::array<OutType, N>{};
        if constexpr (std::is_unsigned_v<OutType>)
        {
            // Unsigned OutType will be truncated below 0: all negative values will be mapped to 0.
            for (std::int32_t i = 0; i < N; ++i)
                tmp[i] = std::clamp(start + i, 0, max_value);
        }
        else
        {
            for (std::int32_t i = 0; i < N; ++i)
                tmp[i] = std::min(start + i, max_value);
        }

        return tmp;
    }
}

#undef XXX_NAMESPACE
