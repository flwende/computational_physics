#pragma once

#include <cstdint>
#include <thread>

#include "simd/simd.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    enum class DeviceType : std::int32_t
    {
        CPU = 1
    };

    template <DeviceType T>
    struct Device;

    class AbstractDevice
    {

        public:
            virtual ~AbstractDevice() = default;

            virtual bool IsOffloadDevice() const = 0;

            virtual std::int32_t Concurrency() const = 0;
    };

    class CPU final : public AbstractDevice
    {
        public:
            CPU() : CPU(std::thread::hardware_concurrency()) {}

            CPU(const std::int32_t concurrency)
                :
                concurrency(concurrency)
            {}

            bool IsOffloadDevice() const override { return false; }

            std::int32_t Concurrency() const override { return concurrency; }

            static constexpr DeviceType Type() { return DeviceType::CPU; }

            template <typename T>
            static constexpr std::int32_t WavefrontSize() { return simd::Type<T>::width; }

        private:
            const std::int32_t concurrency;
    };

    template <>
    struct Device<DeviceType::CPU>
    {
        using Type = CPU;
    };
}