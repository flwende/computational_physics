#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "array/multi_dimensional_array.hpp"
#include "device/device.hpp"
#include "thread_group/thread_group.hpp"

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

            template <template <template <DeviceName> typename, DeviceName> typename Strategy,
                template <DeviceName> typename RNG, DeviceName Target = DeviceName::CPU>
            void Update(const float temperature)
            {
                using StrategyImplementation = Strategy<RNG, Target>;
                static_assert(StrategyImplementation::Dimension == Dimension, "Error: Dimension mismatch.");

                if constexpr (Target == DeviceName::CPU)
                    CreateThreadGroup(); // Create thread group if not done already.

                static StrategyImplementation strategy(*thread_group);
                strategy.Update(*this, temperature);
            }

            template <DeviceName Target = DeviceName::CPU>
            std::pair<double, double> GetEnergyAndMagnetization();

        protected:
            void CreateThreadGroup()
            {
                if (!thread_group.get())
                {
                    const std::int32_t num_threads = GetEnv("NUM_THREADS", std::thread::hardware_concurrency());
                    thread_group.reset(new ThreadGroup(num_threads));
                }
            }

            const std::array<std::int32_t, Dimension> extent;
            const std::size_t num_sites;
            MultiDimensionalArray<Spin, Dimension> spins;
            std::unique_ptr<ThreadGroup> thread_group;

    };
}

#undef XXX_NAMESPACE
