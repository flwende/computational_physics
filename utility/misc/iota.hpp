#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <typename T, std::int32_t N>
    constexpr auto Iota(const T start, const T max_value = std::numeric_limits<T>::max())
    {
        static_assert(N >= 0, "Negative N is not allowed.");

        auto tmp = std::array<T, N>{};
        if constexpr (std::is_unsigned_v<T>)
        {
            for (std::int32_t i = 0; i < N; ++i)
                tmp[i] = std::max(T{0}, std::min(start + i, max_value));
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
