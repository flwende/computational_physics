#pragma once

#include <cassert>
#include <cstdint>
#include <set>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <typename T>
    const T& GetElement(const std::set<T>& data, const std::uint32_t index)
    {
        assert(index < data.size() && "Out of range data access.");

        auto it = std::begin(data);
        std::advance(it, index);
        
        return *it;
    }
}

#undef XXX_NAMESPACE
