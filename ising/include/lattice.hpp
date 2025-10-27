#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "array/multi_dimensional_array.hpp"
#include "device/device.hpp"
#include "future/future.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::uint32_t Dimension, template <DeviceName> typename RNG, DeviceName Target>
    class LatticeMonteCarloAlgorithm;

    template <std::uint32_t Dimension>
    class Lattice final
    {
        public:
            using Spin = std::uint8_t;

        private:
            const std::array<std::uint32_t, Dimension> extent {};
            const std::size_t num_sites {};
            MultiDimensionalArray<Spin, Dimension> spins {};
#if defined __HIPCC__
            GpuPointer<Spin> gpu_spins {};
#endif

        public:
            Lattice(const std::array<std::uint32_t, Dimension>& extent);

            auto& Extent() const noexcept { return extent; }
            auto NumSites() const noexcept { return num_sites; }

            auto RawPointer() const noexcept { return spins.RawPointer(); }
            auto RawPointer() noexcept { return spins.RawPointer(); }

            auto operator[](const std::int32_t index) { return spins[index]; }
            auto operator[](const std::int32_t index) const { return spins[index]; }

            template <template <template <DeviceName> typename, DeviceName> typename Strategy,
                template <DeviceName> typename RNG, typename Target,
                typename StrategyImplementation = Strategy<RNG, Target::Name()>>
            requires std::derived_from<Target, AbstractDevice> &&
                std::derived_from<StrategyImplementation, LatticeMonteCarloAlgorithm<Dimension, RNG, Target::Name()>>
            void Update(const float temperature, Target& target)
            {
                if constexpr (Target::Name() == DeviceName::AMD_GPU)
                    InitializeGpuSpins(target);
                
                static auto strategy = StrategyImplementation(target);
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
            Future<std::pair<double, double>> GetEnergyAndMagnetization(Target& target, const bool async = false);

            template <DeviceName Target = DeviceName::CPU>
            auto GetEnergyAndMagnetization(const bool async = false)
            {
                return GetEnergyAndMagnetization(GetDevice<Target>(), async);
            }

#if defined __HIPCC__
            auto RawGpuPointer() noexcept { return gpu_spins.get(); }
            auto RawGpuPointer() const noexcept { return gpu_spins.get(); }
#endif

        protected:
            // Return target device using defaults.
            template <DeviceName Target = DeviceName::CPU>
            auto& GetDevice() noexcept
            {
                using DeviceType = typename Device<Target>::Type;
                static auto target = DeviceType{};
                return target;
            }

#if defined __HIPCC__
            void InitializeGpuSpins(AMD_GPU& gpu);
#endif
    };
}

#undef XXX_NAMESPACE
