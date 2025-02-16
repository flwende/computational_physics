#pragma once

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::int32_t Dimension>
    class LatticeMonteCarloAlgorithm
    {
        public:
            virtual ~LatticeMonteCarloAlgorithm() = default;

            virtual void Update(const float temperature) = 0;
    };  
}

#undef XXX_NAMESPACE