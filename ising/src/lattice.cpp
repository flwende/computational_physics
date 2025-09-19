#include <atomic>
#include <cstdlib>
#include <numeric>

#include "environment/environment.hpp"
#include "misc/accumulate.hpp"
#include "lattice.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <std::uint32_t Dimension>
    Lattice<Dimension>::Lattice(const std::array<std::uint32_t, Dimension>& extent)
        :
        extent(extent),
        num_sites(Accumulate<std::multiplies<std::size_t>>(extent, 1UL)),
        spins(extent)
    {
        auto* ptr = RawPointer();

        srand48(1);
        for (std::size_t i = 0; i < num_sites; ++i)
            ptr[i] = drand48() < 0.5 ? 0 : 1;
    }

    template <>
    template <>
    std::pair<double, double> Lattice<2>::GetEnergyAndMagnetization<CPU>(CPU& cpu)
    {
        const auto n_0 = extent[0];
        const auto n_1 = extent[1];

        auto energy = std::atomic<std::int64_t>{0};
        auto magnetization = std::atomic<std::int64_t>{0};

        auto kernel = [&, this] (ThreadContext& context)
            {
                const auto thread_id = context.ThreadId();
                const auto num_threads = context.NumThreads();

                const auto y_chunk = (n_1 + num_threads - 1) / num_threads;
                const auto start = thread_id * y_chunk;
                const auto end = std::min(start + y_chunk, n_1);

                auto e = std::int64_t{0};
                auto m = std::int64_t{0};

                for (std::uint32_t y = start; y < end; ++y)
                {
                    for (std::uint32_t x = 0; x < n_0; ++x)
                    {
                        e += (2 * spins[y][x] - 1) * ((2 * spins[y][(x + 1) % n_0] - 1) + (2 * spins[(y + 1) % n_1][x] - 1));
                        m += (2 * spins[y][x] - 1);
                    }
                }

                energy += e;
                magnetization += m;
            };

        cpu.Execute(kernel);

        return {-1.0 * energy / num_sites, 1.0 * magnetization / num_sites};
    }

#if defined __HIPCC__
    template <>
    template <>
    std::pair<double, double> Lattice<2>::GetEnergyAndMagnetization<AMD_GPU>(AMD_GPU& gpu)
    {
        return {1.0, 0.0};
    }

    template <>
    void Lattice<2>::InitializeGpuSpins(AMD_GPU& gpu)
    {
        if (!gpu_spins.get())
        {
            std::cout << "Initializing GPU spins .. ";

            Spin* ptr{};
            SafeCall(hipSetDevice(gpu.DeviceId()));
            SafeCall(hipMalloc(&ptr, num_sites * sizeof(Spin)));
            SafeCall(hipMemcpy(ptr, spins.RawPointer(), num_sites * sizeof(Spin), hipMemcpyHostToDevice));
            gpu_spins.reset(ptr);

            std::cout << "Done." << std::endl;
        }
    }
#endif

    template class Lattice<2>;
}