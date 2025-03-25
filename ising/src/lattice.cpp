#include <atomic>
#include <cstdlib>
#include <numeric>

#include "environment/environment.hpp"
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
    template <>
    std::pair<double, double> Lattice<2>::GetEnergyAndMagnetization<DeviceName::CPU>()
    {
        const std::int32_t n_0 = extent[0];
        const std::int32_t n_1 = extent[1];

        std::atomic<std::int64_t> energy{0};
        std::atomic<std::int64_t> magnetization{0};
        
        // Create thread group if not done already.
        CreateThreadGroup();

        auto kernel = [&, this] (ThreadContext& context)
            {
                const std::int32_t thread_id = context.ThreadId();
                const std::int32_t num_threads = context.NumThreads();

                const std::int32_t y_chunk = (n_1 + num_threads - 1) / num_threads;
                const std::int32_t start = thread_id * y_chunk;
                const std::int32_t end = std::min(start + y_chunk, n_1);

                std::int64_t e{0}, m{0};

                for (std::int32_t y = start; y < end; ++y)
                {
                    for (std::int32_t x = 0; x < n_0; ++x)
                    {
                        e += (2 * spins[y][x] - 1) * (
                            (2 * spins[y][(x + 1) % n_0] - 1) +
                            (2 * spins[(y + 1) % n_1][x] - 1));
                        m += (2 * spins[y][x] - 1);
                    }
                }

                energy += e;
                magnetization += m;
            };

        thread_group->Execute(kernel);

        return {-1.0 * energy / num_sites, 1.0 * magnetization / num_sites};
    }

    template class Lattice<2>;
}