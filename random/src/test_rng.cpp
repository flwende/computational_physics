#include <iostream>
#include <cstdint>
#include <utility>

#include "device/device.hpp"
#include "random/random.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

constexpr std::size_t WarmupIterations {1UL * 1024 * 1024};
constexpr std::size_t BenchmarkIterations {4UL * 1024 * 1024 * 1024};

constexpr DeviceType Target = DeviceType::CPU;
using RNG = LCG32<Target>;

// Returns elapsed time in seconds for running the benchmark.
template <typename RNG, DeviceType Target>
std::pair<double, std::vector<float>> Benchmark(const std::int32_t reporting_id, const std::pair<std::size_t, std::size_t> iterations);

int main(int argc, char** argv)
{
    const std::int32_t reporting_id = (argc > 1 ? std::atoi(argv[1]) : 0);

    // Benchmark.
    const auto [elapsed_time_s, random_numbers] = Benchmark<RNG, Target>(reporting_id, {WarmupIterations, BenchmarkIterations});

    // Reporting.
    for (const auto& number : random_numbers)
        std::cout << number << " ";
    std::cout << std::endl;

    const double time_per_random_number = (elapsed_time_s / BenchmarkIterations);
    std::cout << "Time per random number = " << time_per_random_number * 1.0e9 << " ns" << std::endl;
    std::cout << "Billion random numbers per second = " << (1.0 / (time_per_random_number * 1.0e9)) << std::endl;

    return 0;
}