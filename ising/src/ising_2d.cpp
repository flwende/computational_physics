#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

namespace defaults
{
    static constexpr std::int32_t n_0 = 32;
    static constexpr std::int32_t n_1 = 32;
    static constexpr float temperature = 2.2691853142130221f;
}

static constexpr std::int32_t n_warmup = 10000;
static constexpr std::int32_t n_sep = 20;
static constexpr std::int32_t n_measurement = n_sep * 100000;

int main(int argc, char **argv)
{
    // Get lattice extent from command line (if there are any arguments).
    const std::int32_t n_0 = (argc > 1 ? atoi(argv[1]) : defaults::n_0);
    const std::int32_t n_1 = (argc > 2 ? atoi(argv[2]) : defaults::n_1);

    // Temperature to simulate at: default is the critical temperatur for the 2-D Ising model.
    const float temperature = (argc > 3 ? atof(argv[3]) : defaults::temperature);

    // Create spin system.
    Lattice<2> lattice({n_0, n_1});
    SwendsenWang_2D<LCG32, DeviceName::CPU> s;

    // Thermalization.
    for (std::int32_t i = 0; i < n_warmup; ++i)
        s.Update(lattice, temperature);

    // Measurement.
    double energy = 0.0;
    double magnetization = 0.0;
    double time = omp_get_wtime();
    {
        for (std::int32_t i = 0; i < n_measurement; i += n_sep)
        {
            for (std::int32_t ii = 0; ii < n_sep; ++ii)
                s.Update(lattice, temperature);

            // Take measurements every n_sep update.
            auto [e, m] = lattice.GetEnergyAndMagnetization();
            energy += e;
            magnetization += m;
        }
    }
    time = omp_get_wtime() - time;

    // Output the update time per site ..
    std::cout << "Update time per site (lattice = " << n_0 << " x " << n_1 << "): ";
    std::cout << time * 1.0e9 / (static_cast<std::int64_t>(n_measurement) * n_0 * n_1) << " ns" << std::endl;

    // .. and mean internal energy and magnetization, both per site.
    energy /= (n_measurement / n_sep);
    std::cout << "Internal energy per site: " << energy << std::endl;
    magnetization /= (n_measurement / n_sep);
    std::cout << "Absolute magnetization per site: " << magnetization << std::endl;

    return 0;
}
