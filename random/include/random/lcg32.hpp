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

    template <std::uint32_t WaveFrontSize>
    class LCG32_State
    {
        template <DeviceName> friend class LCG32;

        protected:
            // Internal state of simd_width many concurrent lcgs.
            std::uint32_t state[WaveFrontSize] {};
            // Lcg parameters a (see NUMERICAL RECIPES).
            std::uint32_t a[WaveFrontSize] {};
            // Lcg parameters c (see NUMERICAL RECIPES).
            std::uint32_t c[WaveFrontSize] {};

            // Number of updates of the internal state already performed.
            std::uint32_t iteration {};

            // Number of updates of the internal state after which the concurrent lcgs exchange their parameters.
            static constexpr auto ShuffleDistance = std::uint32_t{15};

        public:
            LCG32_State(std::uint32_t seed = 1) noexcept { Init(seed); }

            void Init(const std::uint32_t seed = 1) noexcept;

            void Update() noexcept;

            auto operator[](const std::int32_t index) const { return state[index]; }

            static constexpr auto GetShuffleDistance() { return ShuffleDistance; }

            // AMD GPU specific functions.
            #include "random/lcg32_amd_gpu.hpp"
    };

    template <>
    class alignas(64) LCG32<DeviceName::CPU> : public RandomNumberGenerator
    {
        static constexpr auto WaveFrontSize = CPU::WavefrontSize<std::uint32_t>();

        public:
            using State = LCG32_State<WaveFrontSize>;

        protected:
            State state {};
            std::uint32_t current {0}; // Current element in the buffer to be accessed next.

        public:
            LCG32(std::uint32_t seed = 1) noexcept : state(seed) {}

            LCG32(State& state) noexcept : state(state) {}

            void Init(const std::uint32_t seed = 1) noexcept override { state.Init(seed); };

            std::uint32_t NextInteger() noexcept override;

            float NextReal() noexcept override;

            void NextInteger(std::vector<std::uint32_t>& numbers) noexcept override
            {
                NextInteger(numbers.data(), numbers.size());
            }

            void NextReal(std::vector<float>& numbers) noexcept override
            {
                NextReal(numbers.data(), numbers.size());
            }

        protected:
            virtual void NextInteger(std::uint32_t* ptr, const std::size_t n) noexcept;

            virtual void NextReal(float* ptr, const std::size_t n) noexcept;
    };

#if defined __HIPCC__
    template <>
    class alignas(128) LCG32<DeviceName::AMD_GPU> : public RandomNumberGenerator
    {
        static constexpr auto WaveFrontSize = AMD_GPU::WavefrontSize();

        public:
            using State = LCG32_State<WaveFrontSize>;
    };
#endif
}

#undef XXX_NAMESPACE
