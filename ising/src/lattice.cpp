#include <atomic>
#include <cstdlib>
#include <memory>
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
        num_sites(Accumulate<std::multiplies<std::size_t>>(extent, std::size_t{1})),
        spins(extent)
    {
        auto* ptr = RawPointer();

        std::srand(1);
        for (std::size_t i = 0; i < num_sites; ++i)
            ptr[i] = std::rand() % 2;
    }

    template <>
    template <>
    Future<std::pair<double, double>> Lattice<2>::GetEnergyAndMagnetization<CPU>(CPU& cpu, const bool async)
    {
        auto energy_magnetization = std::make_shared<std::pair<double, double>>(0, 0);

        auto kernel = [this, energy_magnetization, n_0 = extent[0], n_1 = extent[1]] (ThreadContext& context)
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

                auto energy = std::atomic_ref<double>(energy_magnetization->first);
                auto magnetization = std::atomic_ref<double>(energy_magnetization->second);

                energy += (-1.0 * e / num_sites);
                magnetization += (1.0 * m / num_sites);
            };

        auto awaitable = (async ? cpu.AsyncExecute(kernel) : cpu.Execute(kernel));

        return {std::move(energy_magnetization), awaitable};
    }

#if defined(__HIPCC__)
    template <>
    template <>
    Future<std::pair<double, double>> Lattice<2>::GetEnergyAndMagnetization<AMD_GPU>(AMD_GPU& gpu, const bool async)
    {
        SafeCall(hipSetDevice(gpu.DeviceId()));

        SafeCall(hipMemcpy(RawPointer(), RawGpuPointer(), NumSites() * sizeof(Spin), hipMemcpyDeviceToHost));

        return GetEnergyAndMagnetization<CPU>(gpu.Host(), async);
    }

    template <>
    void Lattice<2>::InitializeGpuSpins(AMD_GPU& gpu)
    {
        if (!gpu_spins.get())
        {
            std::cout << "Initializing GPU spins .. ";

            Spin* ptr{};
            SafeCall(hipSetDevice(gpu.DeviceId()));
            SafeCall(hipMalloc(&ptr, NumSites() * sizeof(Spin)));
            SafeCall(hipMemcpy(ptr, RawPointer(), NumSites() * sizeof(Spin), hipMemcpyHostToDevice));
            gpu_spins.reset(ptr);

            std::cout << "Done." << std::endl;
        }
    }
#endif

    template class Lattice<2>;
}