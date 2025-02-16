#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

#include "random/random.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

#define WARMUP (1024 * 1024)
#define MEASUREMENT (1 * 1024 * 1024 * 1024)

int main()
{
    #pragma omp parallel
    {
        constexpr std::size_t buffer_size = 32;
        std::vector<float> numbers(buffer_size);
        XXX_NAMESPACE::LCG32 rng(1 + omp_get_thread_num());

        for (std::size_t i = 0; i < (WARMUP / buffer_size); ++i)
        {
            rng.NextReal(numbers);
        }

        #pragma omp barrier

        double time = omp_get_wtime();

        #pragma omp barrier

        for (std::size_t i = 0; i < (MEASUREMENT / buffer_size); ++i)
        {
            rng.NextReal(numbers);
        }

        #pragma omp barrier

        time = omp_get_wtime() - time;

        #pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num() << ": " << std::endl;
            for (std::size_t i = 0; i < buffer_size; ++i)
            {
                std::cout << numbers[i] << " ";
            }
            std::cout << std::endl;
        }

        #pragma omp barrier

        #pragma omp master
        {
            const double time_per_random_numbers = ((time / omp_get_num_threads()) / MEASUREMENT);
            std::cout << "Time per random number = " << time_per_random_numbers * 1.0e9 << " ns" << std::endl;
            std::cout << "Billion random numbers per second = " << (1.0 / time_per_random_numbers) * 1.0e-9 << std::endl;
        }
    }

    return 0;
}
