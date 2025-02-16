#pragma once

#include <cstdlib>

#include "random.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Implementation of class random using the 32-bit linear congruential generator (lcg)
    // a la NUMERICAL RECIPES.
    //
    // Reference:
    //
    // * Saul Teukolsky, William H. Press and William T. Vetterling,
    //      "Numerical Recipes in C: The Art of Scientific Computing, 3rd Edition"
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class LCG32 : public RandomNumberGenerator
    {
        public:
            LCG32(std::uint32_t seed = 1);

            void Init(std::uint32_t seed) override;

            std::uint32_t NextInteger() override;

            float NextReal() override;

            void NextInteger(std::vector<std::uint32_t>& numbers) override
            {
                NextInteger(numbers.data(), numbers.size());
            }

            void NextReal(std::vector<float>& numbers) override
            {
                NextReal(numbers.data(), numbers.size());
            }

        protected:
            // Update the internal state: in case of RANDOM_SHUFFLE_STATE is defined,
            // the lcg states are exchanged every shuffle_distance-th updates.
            void Update(std::uint32_t* ptr = nullptr);

            virtual void NextInteger(std::uint32_t* ptr, const std::size_t n);

            virtual void NextReal(float* ptr, const std::size_t n);

            // SIMD width for data type std::uint32_t on the selected platform.
            static constexpr std::int32_t simd_width = simd::Type<std::uint32_t>::width;
            // Internal state of simd_width many concurrent lcgs.
            std::uint32_t state[simd_width];
            // Lcg parameters a (see NUMERICAL RECIPES).
            std::uint32_t a[simd_width];
            // Lcg parameters c (see NUMERICAL RECIPES).
            std::uint32_t c[simd_width];

            // Internal buffer size: we generate chunks of random numbers (multiple of simd_width).
            static constexpr std::int32_t buffer_size = 4 * simd_width;
            // Internal buffer.
            std::uint32_t buffer[buffer_size];
            // Current element in the buffer to be accessed next.
            std::int32_t current_element;
            // Number of updates of the internal state already performed.
            std::uint32_t num_updates;

            // Number of updates of the internal state after which the concurrent lcgs exchange their parameters.
            static constexpr std::size_t shuffle_distance = 15;
    };
}

#undef XXX_NAMESPACE
