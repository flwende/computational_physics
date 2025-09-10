#include <algorithm>
#include <cmath>

#include "environment/environment.hpp"
#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <template <DeviceName> typename RNG, DeviceName Target>
    SwendsenWang_2D<RNG, Target>::SwendsenWang_2D(DeviceType& target)
        :
        target(target)
    {
        const std::int32_t num_threads = target.Concurrency();

        rng_state.reserve(num_threads);
        for (std::int32_t i = 0; i < num_threads; ++i)
            rng_state.emplace_back(i + 1);

        if constexpr (Target == DeviceName::CPU)
        {
            rng.reserve(num_threads);
            for (std::int32_t i = 0; i < num_threads; ++i)
                rng.emplace_back(new RNG<Target>(rng_state[i]));
        }
#if defined __HIPCC__
        else if constexpr (Target == DeviceName::AMD_GPU)
        {
            InitializeGpuRngState();
        }
#endif
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::Update(Lattice<2>& lattice, const float temperature)
    {
        if (!cluster.Initialized())
            cluster.Resize(lattice.Extent());

#if defined __HIPCC__
        if constexpr (Target == DeviceName::AMD_GPU)
        {
            InitializeGpuRngState();
            InitializeGpuCluster(lattice.NumSites());
        }
#endif

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
#if defined __HIPCC__
    template class SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>;
#endif
}