#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "array/multi_dimensional_array.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::int32_t Dimension>
    class Lattice final
    {
        using Spin = std::uint8_t;

        public:
            Lattice(const std::array<std::int32_t, Dimension>& extent);

            const auto& Extent() const { return extent; }
            auto NumSites() const { return num_sites; }

            const auto RawPointer() const { return spins.RawPointer(); }
            auto RawPointer() { return spins.RawPointer(); }

            auto operator[](const std::int32_t index) { return spins[index]; }
            const auto operator[](const std::int32_t index) const { return spins[index]; }

            std::pair<double, double> GetEnergyAndMagnetization() const;

        protected:
            const std::array<std::int32_t, Dimension> extent;
            const std::size_t num_sites;
            MultiDimensionalArray<Spin, Dimension> spins;
    };
}
