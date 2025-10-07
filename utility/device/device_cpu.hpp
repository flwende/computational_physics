#pragma once

#include <cstdint>
#include <thread>

#include "environment/environment.hpp"
#include "execution/thread_group.hpp"
#include "simd/simd.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class CPU final : public AbstractDevice
    {
        private:
            const std::uint32_t concurrency {};
            ThreadGroup thread_group;

        public:
            CPU() : CPU(GetEnv("NUM_THREADS", std::thread::hardware_concurrency()), 0 /* device_id */) {}

            CPU(const std::uint32_t concurrency, const std::uint32_t device_id)
                :
                AbstractDevice(DeviceName::CPU, device_id), concurrency(concurrency), thread_group(concurrency)
            {}

            bool IsOffloadDevice() const noexcept override { return false; }

            std::uint32_t Concurrency() const noexcept override { return concurrency; }

            static constexpr auto Name() noexcept { return DeviceName::CPU; }

            template <typename T>
            static constexpr auto WavefrontSize() noexcept { return simd::Type<T>::Width; }

            template <typename Func, typename ...Args>
            void Execute(Func&& func, Args&&... args)
            {
                thread_group.Execute(std::forward<Func>(func), std::forward<Args>(args)...);
            }

            void SetManagedStackMemorySize(const std::uint32_t bytes)
            {
                thread_group.SetManagedStackMemorySize(bytes);
            }

            void Synchronize()
            {
                thread_group.Synchronize();
            }
    };

    template <>
    struct Device<DeviceName::CPU>
    {
        using Type = CPU;
    };
}

#undef XXX_NAMESPACE
