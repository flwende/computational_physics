#pragma once

#include "environment/environment.hpp"
#include "hip/hip.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
#if defined __HIPCC__
    class AMD_GPU final : public AbstractDevice
    {
        public:
            AMD_GPU(const std::uint32_t device_id = 0) 
                : 
                AMD_GPU(GetEnv("NUM_THREADS", HipGetDeviceProperty("multiProcessorCount", device_id)), device_id)
            {}

            AMD_GPU(const std::uint32_t concurrency, const std::uint32_t device_id)
                :
                AbstractDevice(DeviceName::AMD_GPU, device_id),
                concurrency(concurrency)
            {}

            bool IsOffloadDevice() const override { return true; }

            std::uint32_t Concurrency() const override { return concurrency; }

            static constexpr DeviceName Name() { return DeviceName::AMD_GPU; }

            template <typename T>
            static constexpr std::int32_t WavefrontSize() { return 64; }

        private:
            const std::uint32_t concurrency;
    };

    template <>
    struct Device<DeviceName::AMD_GPU>
    {
        using Type = AMD_GPU;
    };
#else
    template <bool HipEnabled = false>
    class AMD_GPU final
    {
        static_assert(HipEnabled, "HIP not enabled");
    };
#endif
}

#undef XXX_NAMESPACE
