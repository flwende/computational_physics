/*
Target name: CPU
WARMUP: 1000 sweeps
MEASUREMENT: 100000 sweeps
Update time per site (lattice = 128 x 128): 3.36697 ns
Internal energy per site: -1.41921
Absolute magnetization per site: -0.324382
*/

#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

static constexpr std::int32_t n_warmup = 1000;
//static constexpr std::int32_t n_warmup = 1;
static constexpr std::int32_t n_sep = 10;
//static constexpr std::int32_t n_sep = 1;

namespace defaults
{
    static constexpr std::int32_t n_0 = 32;
    static constexpr std::int32_t n_1 = 32;
    static constexpr float temperature = 2.2691853142130221f;
    static constexpr std::int32_t n_sweeps = n_sep * 10000;
}

int main(int argc, char **argv)
{
    // Initialize defaults
    std::int32_t n_0 {defaults::n_0};
    std::int32_t n_1 {defaults::n_1};
    float temperature {defaults::temperature};
    std::int32_t n_sweeps {defaults::n_sweeps};
    std::string algorithm {"swendsen_wang"};
    std::string rng_name {"lcg32"};
    std::string target_name {"cpu"};

    // Display help if requested.
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  --help, -h         Show this help message" << std::endl
                  << "  --extent=N0xN1     Set lattice dimensions" << std::endl
                  << "  --num_sweeps       Set number of MC sweeps (default: 100000)" << std::endl
                  << "  --temperature      Set temperature (default: 2.2691853)" << std::endl
                  << "  --algorithm        Set algorithm (default: swendsen_wang)" << std::endl
                  << "  --rng=<name>       Set RNG type (default: lcg32)" << std::endl
                  << "  --target=<name>    Set target device (default: cpu)" << std::endl;
        return 0;
    }

    // Parse command line arguments in format --key=value.
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (const auto pos = arg.find('='); pos != std::string::npos)
        {
            const std::string key = arg.substr(0, pos);
            const std::string value = arg.substr(pos + 1);
            if (key == "--extent")
            {
                if (const auto x_pos = value.find('x'); x_pos != std::string::npos)
                {
                    n_0 = std::atoi(value.substr(0, x_pos).c_str());
                    n_1 = std::atoi(value.substr(x_pos + 1).c_str());
                }
            }
            else if (key == "--num_sweeps")
                n_sweeps = std::atoi(value.c_str());
            else if (key == "--temperature")
                temperature = std::atof(value.c_str());
            else if (key == "--algorithm")
                algorithm = value;
            else if (key == "--rng")
                rng_name = value;
            else if (key == "--target")
                target_name = value;
        }
    }

    // Create spin system.
    Lattice<2> lattice({n_0, n_1});

    // Target device.
    auto target = GetDevice(target_name);
    std::cout << "Target name: " << (target->Name() == DeviceName::CPU ? "CPU" : "AMD GPU") << std::endl;

    // Update kernel: resolve parameters and set up call.
    auto kernel = [&] <typename DeviceType> (DeviceType& target, const std::int32_t iterations)
        {
            if (rng_name == "lcg32")
            {
                if (algorithm == "swendsen_wang")
                {
                    for (std::int32_t i = 0; i < iterations; ++i)
                        lattice.Update<SwendsenWang_2D, LCG32>(temperature, target);
                }
                else
                    throw std::runtime_error("Unknown algorithm.");
            }
            else
                throw std::runtime_error("Unknown RNG name.");
        };

    std::cout << "WARMUP: " << n_warmup << " sweeps" << std::endl;
    std::cout << "MEASUREMENT: " << n_sweeps << " sweeps" << std::endl;

    // Thermalization.
    DispatchCall(target_name, *target, kernel, n_warmup);

    // Measurement.
    double energy = 0.0;
    double magnetization = 0.0;
    const auto starttime = std::chrono::high_resolution_clock::now();
    {
        for (std::int32_t i = 0; i < n_sweeps; i += n_sep)
        {
            // Take measurements every n_sep update steps.
            DispatchCall(target_name, *target, kernel, n_sep);
            
            auto [e, m] = DispatchCall(target_name, *target, [&lattice] (auto& target)
                {
                    return lattice.GetEnergyAndMagnetization(target);
                });

            energy += e;
            magnetization += m;
        }
    }
    const auto endtime = std::chrono::high_resolution_clock::now();
    const double elapsed_time_s = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0e-6;

    // Output the update time per site ..
    std::cout << "Update time per site (lattice = " << n_0 << " x " << n_1 << "): ";
    std::cout << elapsed_time_s * 1.0e9 / (static_cast<std::int64_t>(n_sweeps) * n_0 * n_1) << " ns" << std::endl;

    // .. and mean internal energy and magnetization, both per site.
    energy /= (n_sweeps / n_sep);
    std::cout << "Internal energy per site: " << energy << std::endl;
    magnetization /= (n_sweeps / n_sep);
    std::cout << "Absolute magnetization per site: " << magnetization << std::endl;

    return 0;
}
