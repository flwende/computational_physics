#include "random/lcg32.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    LCG32::LCG32(std::uint32_t seed)
        :
        current_element(buffer_size - 1),
        num_updates(0)
    {
        // Prameters are taken from NUMERICAL RECIPES.
        static constexpr std::uint32_t num_parameters = 5;
        static constexpr std::uint32_t parameters[num_parameters][2] = {
            {1372383749u, 1289706101u},
            {2891336453u, 1640531513u},
            {2024337845u, 797082193u},
            {32310901u, 626627237u},
            {29943829u, 1013904223u}};

        std::uint32_t seed_init;
        #pragma omp critical (RANDOM_LOCK)
        {
            // Random assignment of parameters to concurrent lcgs.
            srand48(seed);
            for (std::int32_t i = 0; i < simd_width; ++i)
            {
                const std::int32_t selection = static_cast<std::int32_t>(1000.0 * drand48()) % num_parameters;
                a[i] = parameters[selection][0];
                c[i] = parameters[selection][1];
            }

            // Generate the seed value for calling Init method.
            seed_init = static_cast<std::uint32_t>(0xEFFFFFFFu * drand48()) + 1;
        }

        Init(seed_init);
    }

    void LCG32::Init(std::uint32_t seed)
    {
        #pragma omp critical (RANDOM_LOCK)
        {
            // Random initialization of the first lcg.
            srand48(seed);
            state[0] = a[0] * (static_cast<std::uint32_t>(0xEFFFFFFFu * drand48()) + 1) + c[0];
        }

        // The n-th lcg is initialized using the state of the (n - 1)-th lcg.
        for (std::int32_t i = 1; i < simd_width; ++i)
            state[i] = a[i] * state[i - 1] + c[i];
    }

    std::uint32_t LCG32::NextInteger()
    {
        if ((++current_element) == buffer_size)
        {
            // Buffer is empty: refill it.
            Update();
            current_element = 0;
        }

        return buffer[current_element];
    }

    float LCG32::NextReal()
    {
        // Convert integer to float over [0.0, 1.0].
        return 2.3283064370807974e-10f * NextInteger();
    }

    void LCG32::NextInteger(std::uint32_t* ptr, const std::size_t n)
    {
        // Write as many chunks of numbers as possible to ptr.
        const std::size_t i_max = (n / buffer_size) * buffer_size;
        for (std::size_t i = 0; i < i_max; i += buffer_size)
            Update(&ptr[i]);

        // Take the rest from the internal buffer.
        for (std::size_t i = i_max; i < n; ++i)
            ptr[i] = NextInteger();
    }

    void LCG32::NextReal(float* ptr, const std::size_t n)
    {
        // Reinterpret the output buffer type and fill with integers.
        std::uint32_t* i_ptr = reinterpret_cast<std::uint32_t*>(ptr);
        NextInteger(i_ptr, n);

        // Convert integer to float over [0.0, 1.0].
        for (std::size_t i = 0; i < n; ++i)
            ptr[i] = 2.3283064370807974e-10f * i_ptr[i];
    }

    void LCG32::Update(std::uint32_t* ptr)
    {
        // If no output buffer is specified, use the internal buffer.
        if (ptr == nullptr)
            ptr = buffer;

        #if defined(RANDOM_SHUFFLE_STATE)
        // Exchange lcg states at random,
        if (((++num_updates) % shuffle_distance) == 0)
        {
            constexpr std::int32_t m = simd_width - 1;
            const std::int32_t shuffle_val = state[num_updates & m] + (num_updates & 1 ? 0 : 1);

            #pragma omp simd
            for (std::int32_t ii = 0; ii < simd_width; ++ii)
                ptr[ii] = a[(ii + shuffle_val) & m];

            #pragma omp simd
            for (std::int32_t ii = 0; ii < simd_width; ++ii)
                a[ii] = ptr[ii];

            #pragma omp simd
            for (std::int32_t ii = 0; ii < simd_width; ++ii)
                ptr[ii] = c[(ii + shuffle_val) & m];

            #pragma omp simd
            for (std::int32_t ii = 0; ii < simd_width; ++ii)
                c[ii] = ptr[ii];
        }
        #endif

        // Generate buffer_size many random integers.
        for (std::int32_t i = 0; i < buffer_size; i += simd_width)
        {
            #pragma omp simd
            for (std::int32_t ii = 0; ii < simd_width; ++ii)
            {
                // Update internal state.
                state[ii] = a[ii] * state[ii] + c[ii];
                // Push to output buffer.
                ptr[i + ii] = state[ii];
            }
        }
    }
}