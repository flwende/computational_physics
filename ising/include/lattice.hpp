#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "array/multi_dimensional_array.hpp"
#include "device/device.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::int32_t Dimension, template <DeviceName> typename RNG, DeviceName Target>
    class LatticeMonteCarloAlgorithm;

    template <std::int32_t Dimension>
    class Lattice final
    {
        public:
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
                template <DeviceName> typename RNG, typename Target,
                typename StrategyImplementation = Strategy<RNG, Target::Name()>>
            requires std::derived_from<Target, AbstractDevice> &&
                std::derived_from<StrategyImplementation, LatticeMonteCarloAlgorithm<Dimension, RNG, Target::Name()>>
            void Update(const float temperature, Target& target)
            {
                if constexpr (Target::Name() == DeviceName::AMD_GPU)
                    InitializeGpuSpins(target);
                
                static StrategyImplementation strategy(target);
                strategy.Update(*this, temperature);
            }

            template <template <template <DeviceName> typename, DeviceName> typename Strategy,
                template <DeviceName> typename RNG, DeviceName Target = DeviceName::CPU>
            void Update(const float temperature)
            {
                Update<Strategy, RNG, Target>(temperature, GetDevice<Target>());
            }

            template <typename Target>
            requires std::derived_from<Target, AbstractDevice>
            std::pair<double, double> GetEnergyAndMagnetization(Target& target);

            template <DeviceName Target = DeviceName::CPU>
            std::pair<double, double> GetEnergyAndMagnetization()
            {
                return GetEnergyAndMagnetization(GetDevice<Target>());
            }

#if defined __HIPCC__
            auto RawGpuPointer() { return gpu_spins.get(); }
            auto RawGpuPointer() const { return gpu_spins.get(); }
#endif

        protected:
            // Return target device using defaults.
            template <DeviceName Target = DeviceName::CPU>
            auto& GetDevice()
            {
                using DeviceType = typename Device<Target>::Type;
                static DeviceType target;
                return target;
            }

            const std::array<std::int32_t, Dimension> extent;
            const std::size_t num_sites;
            MultiDimensionalArray<Spin, Dimension> spins;

#if defined __HIPCC__
            GpuPointer<Spin> gpu_spins;
            void InitializeGpuSpins(AMD_GPU& gpu);
#endif
    };
}

#undef XXX_NAMESPACE
