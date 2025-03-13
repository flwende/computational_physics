#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "device/device.hpp"
#include "random/random.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

// Don't use LockFreeBarrier as we have 1 additional observer thread which would spin on the barrier.
static Barrier barrier;

constexpr std::int32_t Buffersize {64};

template <typename RNG>
void Kernel(RNG& rng, std::vector<float>& output, const std::pair<std::size_t, std::size_t>& iterations, bool write_back)
{
    const auto [warmup_iterations, benchmark_iterations] = iterations;
    std::vector<float> random_numbers(Buffersize);

    // WarmupIterations.
    for (std::size_t i = 0; i < (warmup_iterations / Buffersize); ++i)
        rng.NextReal(random_numbers);

    // Synchronize with all other threads before starting
    // the Kernel loop.
    barrier.Wait();

    // Kernel.
    for (std::size_t i = 0; i < (benchmark_iterations / Buffersize); ++i)
        rng.NextReal(random_numbers);

    // Wait for all other threads.
    barrier.Wait();

    if (write_back)
        output.swap(random_numbers);
}

template <typename RNG, DeviceType Target>
std::pair<double, std::vector<float>> Benchmark(const std::int32_t reporting_id, const std::pair<std::size_t, std::size_t> iterations)
{
    const typename Device<Target>::Type target;
    const std::int32_t num_threads = target.Concurrency();
    barrier.Reset(num_threads + 1);

    std::vector<RNG> rng;
    rng.reserve(num_threads);
    for (std::int32_t i = 0; i < num_threads; ++i)
       rng.emplace_back(i + 1);

    const auto [warmup_iterations, benchmark_iterations] = iterations;
    std::vector<float> random_numbers(Buffersize);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::int32_t i = 0; i < num_threads; ++i)
        threads.emplace_back(Kernel<RNG>, std::ref(rng.at(i)), std::ref(random_numbers),
            std::make_pair<std::int32_t, std::int32_t>(warmup_iterations / num_threads, benchmark_iterations / num_threads),
            i == reporting_id);

    // Synchronize with threads in the Kernel kernel.
    barrier.Wait();
    const auto starttime = std::chrono::high_resolution_clock::now();
    barrier.Wait();
    const auto endtime = std::chrono::high_resolution_clock::now();
    const double elapsed_time_s = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0e-6;

    for (auto& thread : threads)
        thread.join();

    return {elapsed_time_s, random_numbers};
}

// Explicit instantiation.
template std::pair<double, std::vector<float>> Benchmark<LCG32<DeviceType::CPU>, DeviceType::CPU>(const std::int32_t, const std::pair<std::size_t, std::size_t>);