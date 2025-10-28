#pragma once

#include <cstdint>
#include <set>
#include <tuple>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    struct CpuCore final
    {
        const std::uint32_t physical_id {};
        const std::uint32_t logical_id {};

        CpuCore() noexcept = default;

        CpuCore(const std::uint32_t physical_id, const std::uint32_t logical_id) noexcept
            :
            physical_id(physical_id), logical_id(logical_id)
        {}

        auto operator<(const CpuCore& other) const noexcept
        {
            return std::tie(physical_id, logical_id) < std::tie(other.physical_id, other.logical_id);
        }
    };

    // Returns a set of physical & logical core mappings.
    std::set<CpuCore> CpuCoreTopology();

    void PinToCpuCore(const std::uint32_t cpu_core);
}

#undef XXX_NAMESPACE
