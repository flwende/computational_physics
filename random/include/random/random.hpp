#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <span>

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
            virtual ~RandomNumberGenerator() noexcept = default;

            virtual void Init(const std::uint32_t seed) noexcept = 0;

            virtual std::uint32_t NextInteger() noexcept = 0;

            virtual float NextReal() noexcept = 0;

            virtual void NextInteger(std::span<std::uint32_t>&& numbers) noexcept
            {
                std::for_each(std::begin(numbers), std::end(numbers),
                    [this] (auto& item) { item = NextInteger(); });
            }

            virtual void NextReal(std::span<float>&& numbers) noexcept
            {
                std::for_each(std::begin(numbers), std::end(numbers),
                    [this] (auto& item) { item = NextReal(); });
            }

            auto NextRealArray() noexcept
            {
                constexpr auto N = simd::Type<float>::Width;
                auto numbers = std::array<float, N>{};

                NextReal(numbers);

                return numbers;
            }
	};	
}

#include "lcg32.hpp"

#undef XXX_NAMESPACE
