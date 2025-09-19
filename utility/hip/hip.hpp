#pragma once

#if defined __HIPCC__

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <hip/hip_runtime.h>

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
}

#undef XXX_NAMESPACE

#endif
