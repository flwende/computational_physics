#pragma once

#if defined __HIPCC__

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <hip/hip_runtime.h>

#include "thread_group/context.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    void SafeCall(hipError_t err);

    std::uint32_t HipGetDeviceProperty(const std::string& property, const std::uint32_t device_id = 0);

    namespace internal
    {
        template <typename T>
        struct GPUMemoryDeleter
        {
            void operator()(T* ptr) const
            {
                if (ptr != nullptr)
                {
                    std::cout << "Delete GPU ptr (0x" << std::hex << reinterpret_cast<std::uintptr_t>(ptr) << ")" << std::endl;
                    SafeCall(hipFree(ptr));
                }
            }
        };
    }

    template <typename T>
    using GpuPointer = std::unique_ptr<T, internal::GPUMemoryDeleter<T>>;

    class AMD_GPU;

    class HipContext final : public Context
    {
        // The meaning of group_size and id is "number of GPUs" and "GPU id".

        public:
            HipContext(const std::int32_t group_size, const std::int32_t id, AMD_GPU& device)
                :
                Context(group_size, id),
                device(device)
            {}

            void Synchronize() override
            {
                SafeCall(hipSetDevice(id));
                SafeCall(hipDeviceSynchronize());
            }

            AMD_GPU& Device() const { return device; }

        private:
            AMD_GPU& device;
    };
}

#define SAFE_CALL(X) XXX_NAMESPACE::SafeCall(X)

#undef XXX_NAMESPACE

#else

#define SAFE_CALL(X)

#endif
