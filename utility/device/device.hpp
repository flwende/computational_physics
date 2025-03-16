#pragma once

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    enum class DeviceName : std::int32_t
    {
        CPU = 1,
        AMD_GPU = 2
    };

    template <DeviceName T>
    struct Device;

    class AbstractDevice
    {
        public:
            virtual ~AbstractDevice() = default;

            virtual bool IsOffloadDevice() const = 0;

            virtual std::uint32_t Concurrency() const = 0;

            DeviceName Name() const { return device_name; }

            std::uint32_t DeviceID() const { return device_id; }

        protected:
            AbstractDevice(const DeviceName device_name, const std::uint32_t device_id)
                :
                device_name(device_name),
                device_id(device_id)
            {}

            DeviceName device_name;
            std::uint32_t device_id;
    };
}

#include "device_cpu.hpp"
#include "device_amd_gpu.hpp"

#undef XXX_NAMESPACE
