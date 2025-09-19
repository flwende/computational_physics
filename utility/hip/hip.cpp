#if defined __HIPCC__

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "hip.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    void SafeCall(hipError_t err)
    {
        if (err != hipSuccess)
        {
            const auto error_message = std::string{hipGetErrorString(err)};
            std::cerr << "HIP Error: " << error_message << std::endl;
            throw std::runtime_error(std::string("HIP error: ") + error_message);
        }
    }

    std::uint32_t HipGetDeviceProperty(const std::string& property, const std::uint32_t device_id)
    {
        hipDeviceProp_t properties {};
        SafeCall(hipGetDeviceProperties(&properties, device_id));

        if (property == "maxThreadsPerBlock")
            return properties.maxThreadsPerBlock;
        else if (property == "totalGlobalMem")
            return properties.totalGlobalMem;
        else if (property == "sharedMemPerBlock")
            return properties.sharedMemPerBlock;
        else if (property == "totalConstMem")
            return properties.totalConstMem;
        else if (property == "multiProcessorCount")
#if defined(__GFX10__) || defined(__GFX11__)
            return 2 * properties.multiProcessorCount; /* RDNA: meaning is 'workgroup processors' not CUs -> multiply by 2 */
#else
            return properties.multiProcessorCount; /* GCN, CDNA: meaning is CUs */
#endif
        else if (property == "maxThreadsPerMultiProcessor")
            return properties.maxThreadsPerMultiProcessor;
        else if (property == "memoryClockRate")
            return properties.memoryClockRate;
        else if (property == "memoryBusWidth")
            return properties.memoryBusWidth;
        else if (property == "major")
            return properties.major;
        else if (property == "minor")
            return properties.minor;
        else if (property == "clockRate")
            return properties.clockRate;
        else if (property == "kernelExecTimeoutEnabled")
            return properties.kernelExecTimeoutEnabled;
        else if (property == "integrated")
            return properties.integrated;
        else if (property == "canMapHostMemory")
            return properties.canMapHostMemory;
        else if (property == "computeMode")
            return properties.computeMode;
        else if (property == "concurrentKernels")
            return properties.concurrentKernels;
        else if (property == "ECCEnabled")
            return properties.ECCEnabled;
        else if (property == "pciBusID")
            return properties.pciBusID;
        else if (property == "pciDeviceID")
            return properties.pciDeviceID;
        else if (property == "tccDriver")
            return properties.tccDriver;
        else if (property == "l2CacheSize")
            return properties.l2CacheSize;
        else if (property == "regsPerBlock")
            return properties.regsPerBlock;
        else if (property == "managedMemory")
            return properties.managedMemory;
        else if (property == "isMultiGpuBoard")
            return properties.isMultiGpuBoard;
        else if (property == "pageableMemoryAccess")
            return properties.pageableMemoryAccess;
        else if (property == "concurrentManagedAccess")
            return properties.concurrentManagedAccess;
        else if (property == "cooperativeLaunch")
            return properties.cooperativeLaunch;
        else if (property == "cooperativeMultiDeviceLaunch")
            return properties.cooperativeMultiDeviceLaunch;
        else if (property == "pageableMemoryAccessUsesHostPageTables")
            return properties.pageableMemoryAccessUsesHostPageTables;
        else if (property == "directManagedMemAccessFromHost")
            return properties.directManagedMemAccessFromHost;
        else if (property == "textureAlignment")
            return properties.textureAlignment;
        else if (property == "texturePitchAlignment")
            return properties.texturePitchAlignment;
        else
            throw std::runtime_error("Unknown property: " + property);
    }
}

#endif
