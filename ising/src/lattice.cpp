#include <cstdlib>
#include <numeric>

#include "lattice.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::int32_t Dimension>
    Lattice<Dimension>::Lattice(const std::array<std::int32_t, Dimension>& extent)
        :
        extent(extent),
        num_sites(std::accumulate(std::begin(extent), std::end(extent), 1, std::multiplies<std::int32_t>())),
        spins(extent)
    {
        auto* ptr = RawPointer();

        srand48(1);
        for (std::size_t i = 0; i < num_sites; ++i)
            ptr[i] = drand48() < 0.5 ? 0 : 1;
    }

    template <>
    std::pair<double, double> Lattice<2>::GetEnergyAndMagnetization() const
    {
        const std::int32_t n_0 = extent[0];
        const std::int32_t n_1 = extent[1];
        std::int64_t energy = 0;
        std::int64_t magnetization = 0;
        
        #pragma omp parallel for schedule(static) reduction(+ : energy, magnetization)
        for (std::int32_t y = 0; y < n_1; ++y)
        {
            for (std::int32_t x = 0; x < n_0; ++x)
            {
                energy += (2 * spins[y][x] - 1) * (
                    (2 * spins[y][(x + 1) % n_0] - 1) +
                    (2 * spins[(y + 1) % n_1][x] - 1));
                magnetization += (2 * spins[y][x] - 1);
            }
        }

        return {-1.0 * energy / num_sites, 1.0 * magnetization / num_sites};
    }

    template class Lattice<2>;
}