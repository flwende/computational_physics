#include <array>
#include <mutex>
#include <immintrin.h>

#include "simd/simd.hpp"
#include "random/lcg32.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
#if defined(__USE_SIMD_INTRINSICS__)
    template <std::uint32_t WaveFrontSize>
    void LCG32_State<WaveFrontSize>::Update() noexcept
    {
        using namespace simd;

        #if defined(RANDOM_SHUFFLE_STATE)
        // Exchange lcg states at random.
        if (((++iteration) % ShuffleDistance) == 0)
        {
            constexpr auto M = WaveFrontSize - 1;
            const auto shuffle_val = state[iteration & M] + (iteration & 1 ? 0 : 1);

            VecStore(&a[0], VecRotateRight<std::uint32_t>(VecLoad(&a[0]), shuffle_val));
            VecStore(&c[0], VecRotateRight<std::uint32_t>(VecLoad(&c[0]), shuffle_val));
        }
        #endif

        // Update internal state.
        const auto vec_a = VecLoad(&a[0]);
        const auto vec_c = VecLoad(&c[0]);
        const auto vec_state = VecLoad(&state[0]);

        VecStore(&state[0], VecAdd<std::uint32_t>(VecMulLo<std::uint32_t>(vec_a, vec_state), vec_c));
    }
#endif

    template class LCG32_State<CPU::WavefrontSize<std::uint32_t>()>;
}
