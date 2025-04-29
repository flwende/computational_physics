
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "device.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    std::unique_ptr<AbstractDevice> GetDevice(const std::string& device_name);

    template <typename Func, typename ...Args>
    auto DispatchCall(const std::string& device_name, AbstractDevice& device, Func&& func, Args&&... args)
    {
        if (device_name == "cpu")
            return func(static_cast<CPU&>(device), std::forward<Args>(args)...);
    #if defined __HIPCC__
        else if (device_name == "amd_gpu")
            return func(static_cast<AMD_GPU&>(device), std::forward<Args>(args)...);
    #endif
        else
            throw std::runtime_error("Unknown device name.");
    }
}

#undef XXX_NAMESPACE
