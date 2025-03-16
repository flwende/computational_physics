#pragma once

#include <thread>

#include "environment/environment.hpp"
#include "simd/simd.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class CPU final : public AbstractDevice
    {
        public:
            CPU() : CPU(GetEnv("NUM_THREADS", std::thread::hardware_concurrency()), 0) {}

            CPU(const std::uint32_t concurrency, const std::uint32_t device_id)
                :
                AbstractDevice(DeviceName::CPU, device_id),
                concurrency(concurrency)
            {}

            bool IsOffloadDevice() const override { return false; }

            std::uint32_t Concurrency() const override { return concurrency; }

            static constexpr DeviceName Name() { return DeviceName::CPU; }

            template <typename T>
            static constexpr std::int32_t WavefrontSize() { return simd::Type<T>::width; }

        private:
            const std::uint32_t concurrency;
    };

    template <>
    struct Device<DeviceName::CPU>
    {
        using Type = CPU;
    };
}

#undef XXX_NAMESPACE
