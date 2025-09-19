#pragma once

#if defined __HIPCC__

#include <cstdint>

#include "hip/hip.hpp"
#include "execution/context.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class AMD_GPU;

    class HipContext final : public Context
    {
        private:
            AMD_GPU& device;

        public:
            // The meaning of group_size and id is "number of GPUs" and "GPU id".
            HipContext(const std::uint32_t group_size, const std::uint32_t id, AMD_GPU& device) noexcept
                :
                Context(group_size, id), device(device)
            {}

            void Synchronize() override
            {
                SafeCall(hipSetDevice(id));
                SafeCall(hipDeviceSynchronize());
            }

            AMD_GPU& Device() const noexcept { return device; }
    };
}

#undef XXX_NAMESPACE

#endif
