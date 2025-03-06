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
    class LCG32;

    template <std::int32_t WaveFrontSize>
    class LCG32_State
    {
        friend class LCG32;

        public:
            LCG32_State(std::uint32_t seed = 1);

            void Init(const std::uint32_t seed = 1);

            void Update();

            auto operator[](const std::int32_t index) const { return state[index]; }

        protected:
            // Internal state of simd_width many concurrent lcgs.
            std::uint32_t state[WaveFrontSize];
            // Lcg parameters a (see NUMERICAL RECIPES).
            std::uint32_t a[WaveFrontSize];
            // Lcg parameters c (see NUMERICAL RECIPES).
            std::uint32_t c[WaveFrontSize];

            // Number of updates of the internal state already performed.
            std::uint32_t iteration;

            // Number of updates of the internal state after which the concurrent lcgs exchange their parameters.
            static constexpr std::size_t shuffle_distance = 15;
    };

    class alignas(128) LCG32 : public RandomNumberGenerator
    {
        public:
            LCG32(std::uint32_t seed = 1);

            void Init(const std::uint32_t seed = 1) override;

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
            virtual void NextInteger(std::uint32_t* ptr, const std::size_t n);

            virtual void NextReal(float* ptr, const std::size_t n);

            // SIMD width for data type std::uint32_t on the selected platform.
            static constexpr std::int32_t WaveFrontSize = simd::Type<std::uint32_t>::width;
            LCG32_State<WaveFrontSize> state;

            // Current element in the buffer to be accessed next.
            std::int32_t current;
    };
}

#undef XXX_NAMESPACE
