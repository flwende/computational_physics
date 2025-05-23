#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "device/device.hpp"
#include "random/random.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

constexpr std::size_t WarmupIterations {1UL * 1024 * 1024};
constexpr std::size_t BenchmarkIterations {4UL * 1024 * 1024 * 1024};

// Returns elapsed time in seconds for running the benchmark.
template <template <DeviceName> typename RNG, DeviceName Target>
std::pair<double, std::vector<float>> Benchmark(const std::int32_t reporting_id, const std::pair<std::size_t, std::size_t> iterations);

// Main program
int main(int argc, char** argv)
{
    std::string rng_name {"lcg32"};
    std::string target_name {"cpu"};
    std::int32_t reporting_id {0};

    // Display help if requested
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  --help, -h      Show this help message" << std::endl
                  << "  --rng=<name>    Set RNG type (default: lcg32)" << std::endl
#if defined __HIPCC__
                  << "  --target=<name> Set target device (default: cpu)" << std::endl
                  << "      Supported target devices: cpu, amd_gpu" << std::endl
#endif
                  << "  --id=<number>   Set reporting ID (default: 0)" << std::endl;
        return 0;
    }

    // Parse command line arguments in format --key=value
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto pos = arg.find('=');
        if (pos != std::string::npos) {
            const std::string key = arg.substr(0, pos);
            const std::string value = arg.substr(pos + 1);
            if (key == "--rng")
                rng_name = value;
            else if (key == "--target")
                target_name = value;
            else if (key == "--id")
                reporting_id = std::stoi(value);
        }
    }

    // Benchmark.
    const auto [elapsed_time_s, random_numbers] = [&] (const std::string rng_name)
        {
            if (rng_name == "lcg32")
            {
                if (target_name == "cpu")
                {
                    return Benchmark<LCG32, DeviceName::CPU>(reporting_id, {WarmupIterations, BenchmarkIterations});
                }
#if defined __HIPCC__
                else if (target_name == "amd_gpu")
                {
                    return Benchmark<LCG32, DeviceName::AMD_GPU>(reporting_id, {WarmupIterations, BenchmarkIterations});
                }
#endif
                else
                    throw std::runtime_error("Unknown target device name.");
            }
            else
            {
                throw std::runtime_error("Unknown RNG name.");
            }
        } (rng_name);

    // Reporting.
    for (const auto& number : random_numbers)
        std::cout << number << " ";
    std::cout << std::endl;

    const double time_per_random_number = (elapsed_time_s / BenchmarkIterations);
    std::cout << "Time per random number = " << time_per_random_number * 1.0e9 << " ns" << std::endl;
    std::cout << "Billion random numbers per second = " << (1.0 / (time_per_random_number * 1.0e9)) << std::endl;

    return 0;
}