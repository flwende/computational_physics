#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "device/device.hpp"
#include "random/random.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

constexpr std::int32_t Buffersize {64};

namespace
{
    template <typename RNG>
    void Kernel(RNG& rng, std::vector<float>& output, const std::size_t iterations, const bool write_back = false)
    {
        std::vector<float> random_numbers(Buffersize);

        for (std::size_t i = 0; i < (iterations / Buffersize); ++i)
            rng.NextReal(random_numbers);

        if (write_back)
            output.swap(random_numbers);
    }
}

template <template <DeviceName> typename RNG, DeviceName Target>
std::pair<double, std::vector<float>> Benchmark(const std::int32_t reporting_id, const std::pair<std::size_t, std::size_t> iterations)
{
    static_assert(Target == DeviceName::CPU, "Target must be CPU.");

    CPU cpu;
    std::vector<float> random_numbers(Buffersize);
    double elapsed_time_s {};

    cpu.Execute([&] (auto& thread_pool)
        {
            const auto [warmup_iterations, benchmark_iterations] = iterations;
            const auto thread_id = thread_pool.ThreadId();
            const auto num_threads = cpu.Concurrency();
            RNG<DeviceName::CPU> rng(thread_id + 1);

            // Warmup.
            Kernel<RNG<DeviceName::CPU>>(rng, random_numbers, warmup_iterations / num_threads);

            thread_pool.Synchronize();
            const auto starttime = std::chrono::high_resolution_clock::now();

            // Benchmark.
            Kernel<RNG<DeviceName::CPU>>(rng, random_numbers, benchmark_iterations / num_threads, thread_id == reporting_id);

            thread_pool.Synchronize();
            const auto endtime = std::chrono::high_resolution_clock::now();

            if (thread_id == 0)
                elapsed_time_s = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0e-6;
        });

    return {elapsed_time_s, random_numbers};
}

// Explicit instantiation.
template std::pair<double, std::vector<float>> Benchmark<LCG32, DeviceName::CPU>(const std::int32_t, const std::pair<std::size_t, std::size_t>);
