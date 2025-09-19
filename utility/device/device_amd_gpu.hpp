#pragma once

#include <cstdint>

#include "environment/environment.hpp"
#include "execution/hip_context.hpp"

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
                AMD_GPU(GetEnv("NUM_THREADS_GPU", HipGetDeviceProperty("multiProcessorCount", device_id)), device_id)
            {}

            AMD_GPU(const std::uint32_t concurrency, const std::uint32_t device_id)
                :
                AbstractDevice(DeviceName::AMD_GPU, device_id),
                concurrency(concurrency),
                context(1 /* number of GPUs */, device_id, *this)
            {}

            bool IsOffloadDevice() const noexcept override { return true; }

            std::uint32_t Concurrency() const noexcept override { return concurrency; }

            static constexpr auto Name() noexcept { return DeviceName::AMD_GPU; }

            template <typename T = void>
            static constexpr std::uint32_t WavefrontSize()
            {
                // Wavefront size is architecture specific.
                #if defined(__GFX10__) || defined(__GFX11__)
                return 32; /* RDNA */
                #else
                return 64; /* GCN, CDNA */
                #endif
            }

            template <typename Func, typename ...Args>
            void Execute(Func&& func, const dim3& grid, const dim3& block, const std::size_t shared_mem_bytes,
                Args&&... args)
            {
                SafeCall(hipSetDevice(DeviceId()));
                hipLaunchKernelGGL(func, grid, block, shared_mem_bytes, 0, std::forward<Args>(args)...);
                //Synchronize();
            }

            template <typename Func, typename ...Args>
            void Execute(Func&& func, Args&&... args)
            {
                SafeCall(hipSetDevice(DeviceId()));
                func(context, std::forward<Args>(args)...);
                //Synchronize();
            }

            void Synchronize()
            {
                context.Synchronize();
            }

            CPU& Host() { return host; }

        private:
            const std::uint32_t concurrency;
            HipContext context;
            CPU host; /* Configured via NUM_THREADS environment variable */
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
