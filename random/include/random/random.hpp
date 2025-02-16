#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "simd/simd.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
	// Random number generator interface.
	class RandomNumberGenerator
	{
        public:
            virtual ~RandomNumberGenerator() = default;

            virtual void Init(std::uint32_t seed) = 0;

            virtual std::uint32_t NextInteger() = 0;

            virtual float NextReal() = 0;

            virtual void NextInteger(std::vector<std::uint32_t>& numbers)
            {
                std::for_each(std::begin(numbers), std::end(numbers),
                    [this] (auto& item) { item = NextInteger(); });
            }

            virtual void NextReal(std::vector<float>& numbers)
            {
                std::for_each(std::begin(numbers), std::end(numbers),
                    [this] (auto& item) { item = NextReal(); });
            }
	};	
}

#include "lcg32.hpp"

#undef XXX_NAMESPACE
