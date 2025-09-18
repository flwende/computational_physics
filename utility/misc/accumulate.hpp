#pragma once

#include <cstdint>
#include <numeric>
#include <ranges>
#include <type_traits>


#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <typename Func, std::ranges::range Data, typename T>
    auto Accumulate(const Data& data, const T seed)
    {
        if constexpr (std::is_same_v<Func, std::multiplies<T>>)
        {
            return std::accumulate(std::begin(data), std::end(data), seed, Func());
        }
        else
        {
            static_assert(false, "Accumulate not implemented for provided function");
        }
    }
}

#undef XXX_NAMESPACE
