// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <omp.h>
#include <random/random.hpp>

#define WARMUP (1024 * 1024)
#define MEASUREMENT (1 * 1024 * 1024 * 1024)

int main()
{
	// simple test program for the lcg

	#pragma omp parallel
	{
		constexpr std::size_t buffer_size = 32;
	 	float numbers[buffer_size];
		fw::lcg32 rng(1 + omp_get_thread_num());

		for (std::size_t i = 0; i < (WARMUP / buffer_size); ++i)
		{
			rng.next_float(numbers, buffer_size);
		}

		#pragma omp barrier
		#pragma omp barrier
		
		double time = omp_get_wtime();

		#pragma omp barrier
		#pragma omp barrier

		for (std::size_t i = 0; i < (MEASUREMENT / buffer_size); ++i)
		{
			rng.next_float(numbers, buffer_size);
		}

		#pragma omp barrier
		#pragma omp barrier
		
		time = omp_get_wtime() - time;

		#pragma omp critical
		{
			std::cout << "thread " << omp_get_thread_num() << ": " << std::endl;
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
			std::cout << "time per random number = " << time_per_random_numbers * 1.0E9 << " ns" << std::endl;
			std::cout << "mrd. random numbers per second = " << (1.0 / time_per_random_numbers) * 1.0E-9 << std::endl;
		}
	}	

	return 0;
}
