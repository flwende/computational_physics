#pragma once

#include <cstdlib>

#include "device/device.hpp"
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
    template <DeviceName Type>
    class LCG32;

    template <std::int32_t WaveFrontSize>
    class LCG32_State
    {
        template <DeviceName> friend class LCG32;

        public:
            LCG32_State(std::uint32_t seed = 1);

            void Init(const std::uint32_t seed = 1);

            void Update();

            auto operator[](const std::int32_t index) const { return state[index]; }

            // AMD GPU specific functions.
            #include "random/lcg32_amd_gpu.hpp"

        protected:
            // Internal state of simd_width many concurrent lcgs.
            std::uint32_t state[WaveFrontSize];
            // Lcg parameters a (see NUMERICAL RECIPES).
            std::uint32_t a[WaveFrontSize];
            // Lcg parameters c (see NUMERICAL RECIPES).
            std::uint32_t c[WaveFrontSize];

            // Number of updates of the internal state already performed.
            std::int32_t iteration;

            // Number of updates of the internal state after which the concurrent lcgs exchange their parameters.
            static constexpr std::int32_t shuffle_distance = 15;
    };

    template <>
    class alignas(128) LCG32<DeviceName::CPU> : public RandomNumberGenerator
    {
        static constexpr std::int32_t WaveFrontSize = CPU::WavefrontSize<std::uint32_t>();

        public:
            using State = LCG32_State<WaveFrontSize>;

            LCG32(std::uint32_t seed = 1);

            LCG32(State& state);

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

            State state;

            // Current element in the buffer to be accessed next.
            std::int32_t current {0};
    };

#if defined __HIPCC__
    template <>
    class alignas(128) LCG32<DeviceName::AMD_GPU> : public RandomNumberGenerator
    {
        static constexpr std::int32_t WaveFrontSize = AMD_GPU::WavefrontSize();

        public:
            using State = LCG32_State<WaveFrontSize>;
    };
#endif
}

#undef XXX_NAMESPACE
