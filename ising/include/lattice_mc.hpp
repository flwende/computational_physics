#pragma once

#include <cstdint>

#include "device/device.hpp"
#include "lattice.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::int32_t Dimension,
        template <DeviceName> typename RNG,
        DeviceName Target>
    class LatticeMonteCarloAlgorithm
    {
        public:
            virtual ~LatticeMonteCarloAlgorithm() = default;

            virtual void Update(Lattice<Dimension>& lattice, const float temperature) = 0;
    };  
}

#undef XXX_NAMESPACE
