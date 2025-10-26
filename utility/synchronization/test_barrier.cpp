#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

static Barrier barrier;

int main(int argc, char** argv)
{
    const auto num_iterations = static_cast<std::uint32_t>(argc > 1 ? std::atoi(argv[1]) : 1);
    const auto num_threads = static_cast<std::uint32_t>(argc > 2 ? std::atoi(argv[2]) : 1);

    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Iteartions: " << num_iterations << std::endl;
    std::cout << "=====================" << std::endl;

    barrier.Reset(num_threads);

    auto kernel = [] (const std::uint32_t num_iterations)
        {
            for (std::uint32_t i = 0; i < num_iterations; ++i)
                barrier.Wait();
        };

    std::vector<std::jthread> threads;
    threads.reserve(num_threads - 1);
    for (std::uint32_t i = 0; i < (num_threads - 1); ++i)
        threads.emplace_back(kernel, num_iterations + 1);
    barrier.Wait();

    // Benchmark.
    auto starttime = std::chrono::high_resolution_clock::now();
    kernel(num_iterations);
    auto endtime = std::chrono::high_resolution_clock::now();

    // Reporting.
    const auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0E-3;
    std::cout << "Time per iteration: " << elapsed_time_ms / num_iterations << "ms" << std::endl;

    return 0;
}