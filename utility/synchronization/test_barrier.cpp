#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

//static Barrier barrier;
static LockFreeBarrier barrier;

int main(int argc, char** argv)
{
    const std::int32_t num_iterations = (argc > 1 ? std::atoi(argv[1]) : 1);
    const std::int32_t num_threads = (argc > 2 ? std::atoi(argv[2]) : 1);

    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Iteartions: " << num_iterations << std::endl;
    std::cout << "=====================" << std::endl;

    barrier.Reset(num_threads);

    auto kernel = [] (const std::int32_t num_iterations)
        {            
            for (std::int32_t i = 0; i < num_iterations; ++i)
                barrier.Wait();
        };

    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1);
    for (std::int32_t i = 0; i < (num_threads - 1); ++i)
        threads.emplace_back(kernel, num_iterations + 1);
    barrier.Wait();

    // Benchmark.
    auto starttime = std::chrono::high_resolution_clock::now();
    kernel(num_iterations);
    auto endtime = std::chrono::high_resolution_clock::now();

    // Reporting.
    const double elapsed_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0e-3;
    std::cout << "Time per iteration: " << elapsed_time_ms / num_iterations << "ms" << std::endl;

    for (auto& thread : threads)
        thread.join();
    
    return 0;
}