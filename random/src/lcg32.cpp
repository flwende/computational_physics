#include <mutex>

#include "random/lcg32.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace
    {
        std::mutex m_init;

        // Prameters are taken from NUMERICAL RECIPES.
        constexpr std::int32_t num_parameters = 5;
        constexpr std::uint32_t parameters[num_parameters][2] = {
            {1372383749u, 1289706101u},
            {2891336453u, 1640531513u},
            {2024337845u, 797082193u},
            {32310901u, 626627237u},
            {29943829u, 1013904223u}};
    }

    template <std::int32_t WaveFrontSize>
    LCG32_State<WaveFrontSize>::LCG32_State(std::uint32_t seed)
        :
        iteration(0)
    {
        Init(seed);
    }

    template <std::int32_t WaveFrontSize>
    void LCG32_State<WaveFrontSize>::Init(const std::uint32_t seed)
    {
        // Random initialization of the first lcg.
        {
            std::lock_guard<std::mutex> lock(m_init);
            srand48(seed);

            // Random assignment of parameters to concurrent lcgs.
            for (std::int32_t i = 0; i < WaveFrontSize; ++i)
            {
                const std::int32_t select = static_cast<std::int32_t>(1000.0 * drand48()) % num_parameters;
                a[i] = parameters[select][0];
                c[i] = parameters[select][1];
            }

            state[0] = a[0] * (static_cast<std::uint32_t>(0xEFFFFFFFu * drand48()) + 1) + c[0];
        }

        // The n-th lcg is initialized using the state of the (n - 1)-th lcg.
        for (std::int32_t i = 1; i < WaveFrontSize; ++i)
            state[i] = a[i] * state[i - 1] + c[i];
    }

    template <std::int32_t WaveFrontSize>
    void LCG32_State<WaveFrontSize>::Update()
    {
        #if defined(RANDOM_SHUFFLE_STATE)
        // Exchange lcg states at random.
        std::uint32_t buffer[WaveFrontSize];
        if (((++iteration) % shuffle_distance) == 0)
        {
            constexpr std::int32_t m = WaveFrontSize - 1;
            const std::int32_t shuffle_val = state[iteration & m] + (iteration & 1 ? 0 : 1);

            #pragma omp simd
            for (std::int32_t i = 0; i < WaveFrontSize; ++i)
                buffer[i] = a[(i + shuffle_val) & m];

            #pragma omp simd
            for (std::int32_t i = 0; i < WaveFrontSize; ++i)
                a[i] = buffer[i];

            #pragma omp simd
            for (std::int32_t i = 0; i < WaveFrontSize; ++i)
                buffer[i] = c[(i + shuffle_val) & m];

            #pragma omp simd
            for (std::int32_t i = 0; i < WaveFrontSize; ++i)
                c[i] = buffer[i];
        }
        #endif

        // Update internal state.
        #pragma omp simd
        for (std::int32_t i = 0; i < WaveFrontSize; ++i)
            state[i] = a[i] * state[i] + c[i];
    }

    LCG32<DeviceType::CPU>::LCG32(std::uint32_t seed)
        :
        state(seed)
    {}

    void LCG32<DeviceType::CPU>::Init(const std::uint32_t seed)
    {
        state.Init(seed);
    }

    std::uint32_t LCG32<DeviceType::CPU>::NextInteger()
    {
        if ((++current) == WaveFrontSize)
        {
            state.Update();
            current = 0;
        }

        return state[current];
    }

    float LCG32<DeviceType::CPU>::NextReal()
    {
        // Convert integer to float over [0.0, 1.0].
        return 2.3283064370807974e-10f * NextInteger();
    }

    void LCG32<DeviceType::CPU>::NextInteger(std::uint32_t* ptr, const std::size_t n)
    {
        const std::size_t i_max = (n / WaveFrontSize) * WaveFrontSize;
        for (std::size_t i = 0; i < i_max; i += WaveFrontSize)
        {
            state.Update();

            #pragma omp simd
            for (std::int32_t ii = 0; ii < WaveFrontSize; ++ii)
                ptr[i + ii] = state[ii];
        }

        current = WaveFrontSize - 1;
        for (std::size_t i = i_max; i < n; ++i)
            ptr[i] = NextInteger();
    }

    void LCG32<DeviceType::CPU>::NextReal(float* ptr, const std::size_t n)
    {
        // Reinterpret the output buffer type and fill with integers.
        std::uint32_t* i_ptr = reinterpret_cast<std::uint32_t*>(ptr);
        NextInteger(i_ptr, n);

        // Convert integer to float over [0.0, 1.0].
        for (std::size_t i = 0; i < n; ++i)
            ptr[i] = 2.3283064370807974e-10f * i_ptr[i];
    }
}
