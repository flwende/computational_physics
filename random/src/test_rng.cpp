#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <pthread.h>

#include "random/random.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

#define WARMUP (1024 * 1024)
#define MEASUREMENT (1 * 1024 * 1024 * 1024)
#define BUFFER_SIZE 32

using namespace XXX_NAMESPACE;
using RNG = LCG32;

static std::mutex io_lock;

// Don't use LockFreeBarrier as we have 1 additional
// observer thread.
static Barrier barrier;

template <typename RNG>
void Benchmark(RNG& rng, std::int32_t id, std::int32_t reporting_id);

int main(int argc, char** argv)
{
    const std::int32_t reporting_id = (argc > 1 ? std::atoi(argv[1]) : 0);

    const std::int32_t num_threads = std::thread::hardware_concurrency();
    barrier.Reset(num_threads + 1);

    std::vector<RNG> rng;
    rng.reserve(num_threads);
    for (std::int32_t i = 0; i < num_threads; ++i)
       rng.emplace_back(i + 1);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::int32_t i = 0; i < num_threads; ++i)
        threads.emplace_back(Benchmark<RNG>, std::ref(rng.at(i)), i, reporting_id);

    // Synchronize with threads in the benchmark kernel.
    barrier.Wait();
    auto starttime = std::chrono::high_resolution_clock::now();
    barrier.Wait();
    auto endtime = std::chrono::high_resolution_clock::now();

    for (auto& thread : threads)
        thread.join();

    // Reporting.
    std::lock_guard<std::mutex> lock(io_lock);
    const double elapsed_time_s = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0e-6;
    const double time_per_random_number = ((elapsed_time_s / num_threads) / MEASUREMENT);
    std::cout << "Time per random number = " << time_per_random_number * 1.0e9 << " ns" << std::endl;
    std::cout << "Billion random numbers per second = " << (1.0 / (time_per_random_number * 1.0e9)) << std::endl;

    return 0;
}

template <typename RNG>
void Benchmark(RNG& rng, std::int32_t id, std::int32_t reporting_id)
{
    std::vector<float> numbers(BUFFER_SIZE);

    // Warmup.
    for (std::size_t i = 0; i < (WARMUP / BUFFER_SIZE); ++i)
        rng.NextReal(numbers);

    // Synchronize with all other threads before starting
    // the benchmark loop.
    barrier.Wait();

    // Benchmark.
    for (std::size_t i = 0; i < (MEASUREMENT / BUFFER_SIZE); ++i)
        rng.NextReal(numbers);

    // Wait for all other threads.
    barrier.Wait();

    if (id == reporting_id)
    {
        std::lock_guard<std::mutex> lock(io_lock);

        std::cout << "Thread " << reporting_id << ": " << std::endl;
        for (std::size_t i = 0; i < BUFFER_SIZE; ++i)
        {
            std::cout << numbers[i] << " ";
        }
        std::cout << std::endl;
    }
}
