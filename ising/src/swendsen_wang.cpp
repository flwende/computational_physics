#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "environment/environment.hpp"
#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace
    {
        template <DeviceName Target, std::size_t Dimension>
        std::array<std::uint32_t, Dimension> GetTileSize(const std::uint32_t wavefront_size, const std::array<std::uint32_t, Dimension>& extent)
        {
            if constexpr (Dimension == 2)
            {
                if constexpr (Target == DeviceName::CPU)
                {
                    return {GetEnv("TILE_SIZE_X", wavefront_size), GetEnv("TILE_SIZE_Y", 8U)};
                }
                else
                {
                    return {2 * wavefront_size / 8, 8};
                }
            }
        }
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    SwendsenWang_2D<RNG, Target>::SwendsenWang_2D(DeviceType& target)
        :
        target(target)
    {}

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::Initialize(Lattice<2>& lattice)
    {
        if (!cluster.Initialized())
        {
            const auto& extent = lattice.Extent();
            if (extent[0] % 2 || extent[1] % 2)
                throw std::runtime_error("SwendsenWang_2D implementation requires evenly sized lattice.");

            std::cout << "Lattice: " << extent[0] << " x " << extent[1] << " (";

            tile_size = GetTileSize<Target>(WavefrontSize, extent);
            std::cout << "tile size: " << tile_size[0] << " x " << tile_size[1] << ")" << std::endl;

            // Resize the cluster ..
            cluster.Resize(extent);

            // .. and initialize random number generators (count depends on the lattice extent).
            const auto concurrency = target.Concurrency();
            const auto num_rngs =  (Target == DeviceName::CPU ? concurrency :
                ((extent[0] + tile_size[0] - 1) / tile_size[0]) * ((extent[1] + tile_size[1] - 1) / tile_size[1]));

            for (std::uint32_t i = 0; i < num_rngs; ++i)
                rng_state.emplace_back(i + 1);

            if constexpr (Target == DeviceName::CPU)
            {
                rng.reserve(num_rngs);
                for (std::uint32_t i = 0; i < num_rngs; ++i)
                    rng.emplace_back(new RNG<Target>(rng_state[i]));

                // Adapt managed stack memory size for cluster update.
                const auto needed_bytes = (2 * tile_size[0] * tile_size[1] + tile_size[0]) * sizeof(LabelType);
                std::cout << "CPU: setting managed stack memory to " << needed_bytes << " bytes" << std::endl;
                target.SetManagedStackMemorySize(needed_bytes);
            }
#if defined(__HIPCC__)
            else if constexpr (Target == DeviceName::AMD_GPU)
            {
                InitializeGpuRngState();
                InitializeGpuCluster(lattice.NumSites());
            }
#endif
            else
            {
                static_assert(false, "Unknown target DeviceName.");
            }
        }
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::Update(Lattice<2>& lattice, const float temperature)
    {
        Initialize(lattice);

        // Probability for adding aligned neighboring sites to the cluster.
        const float p_add = 1.0f - static_cast<float>(std::exp(-2.0f / temperature));
        target.Execute([&,this] (Context& context, auto&&... args)
            {
                AssignLabels(context, args...);
                context.Synchronize();
                MergeLabels(context, args...);
            },
            std::ref(lattice), p_add);

        target.Execute([&,this] (auto&&... args) { ResolveLabels(args...); });

        target.Execute([&,this] (auto&&... args) { FlipClusters(args...); }, std::ref(lattice));
    }

    // Explicit template instantiation.
    template class SwendsenWang_2D<LCG32, DeviceName::CPU>;
#if defined(__HIPCC__)
    template class SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>;
#endif
}