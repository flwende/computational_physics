
#include "helper.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    std::unique_ptr<AbstractDevice> GetDevice(const std::string& device_name)
    {
        if (device_name == "cpu")
            return std::make_unique<CPU>();
#if defined(__HIPCC__)
        else if (device_name == "amd_gpu")
            return std::make_unique<AMD_GPU>();
#endif
        else
            throw std::runtime_error("Unknown target device.");
    }
}

#undef XXX_NAMESPACE
