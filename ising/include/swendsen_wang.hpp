#pragma once

#include <array>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "array/multi_dimensional_array.hpp"
#include "atomic/atomic.hpp"
#include "random/random.hpp"
#include "simd/simd.hpp"
#include "lattice_mc.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Swendsen Wang multi-cluster algorithm for the 2-dimensional Ising model.
    //
    // Reference: R. H. Swendsen and J. S. Wang,
    //            "Nonuniversal critical dynamics in Monte Carlo simulations"
    //            Phys. Rev. Lett., 58:86-88, Jan 1987
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <template <DeviceName> typename RNG, DeviceName Target>
    class SwendsenWang_2D : public LatticeMonteCarloAlgorithm<2, RNG, Target>
    {
        // Set to std::uint64_t if more then 4 billion labels.
        using LabelType = std::uint32_t;

        using DeviceType = typename Device<Target>::Type;

        static constexpr std::uint32_t WavefrontSize = DeviceType::template WavefrontSize<LabelType>();

        public:
            SwendsenWang_2D(DeviceType& target);

            // Sweep (update the entire lattice).
            // Each update comprises calling the following methods:
            //
            // 1. AssignLabels (using e.g. CCL_SelfLabeling)
            //
            // 2. MergeLabels
            //
            // 3. ResolveLabels
            //
            // 4. FlipSpins
            //
            // Steps 1-3: find all clusters (connected components)
            //
            // Sep 4: flip clusters individually
            void Update(Lattice<2>& lattice, const float temperature) override;

            static constexpr std::uint32_t Dimension = 2;

        protected:
            // Connected component labeling (ccl) based on an idea of Coddington and Baillie within tiles.
            template <std::uint32_t N_0 = 0>
            void CCL_SelfLabeling(Context& context, Lattice<2>& lattice, const float p_add, const std::array<uint32_t, 2>& n_offset, const std::array<uint32_t, 2>& n_sub);

            // Loop over all tiles of the lattice and apply ccl_selflabeling.
            // Parameter p_add is the probability for adding aligned nearest
            // neighbor sites to a cluster.
            void AssignLabels(Context& context, Lattice<2>& lattice, const float p_add);

            // Connect all tiles.
            // Parameter p_add is the probability for adding aligned nearest
            // neighbor sites to the cluster.
            void MergeLabels(Context& context, Lattice<2>& lattice, const float p_add);

            // Helper method to establish label equivalences, thus merging clusters
            void Merge(LabelType* ptr, LabelType a, LabelType b);

            // Resolve all label equivalences.
            void ResolveLabels(Context& context);

            // Flip clusters.
            void FlipClusters(Context& context, Lattice<2>& lattice);

            DeviceType& target;
            MultiDimensionalArray<LabelType, 2> cluster;
            const std::array<std::uint32_t, 2> tile_size;

            // Random number generator: if you use the lcg32 generator, make sure you are
            // compiling with RANDOM_SHUFFLE_STATE (otherwise, random numbers might have too low quality).
            using RngState = typename RNG<Target>::State;
            std::vector<RngState> rng_state;
            std::vector<std::shared_ptr<RandomNumberGenerator>> rng;

#if defined __HIPCC__
            GpuPointer<RngState> gpu_rng_state;
            void InitializeGpuRngState();

            GpuPointer<LabelType> gpu_cluster;
            void InitializeGpuCluster(const LabelType num_sites);
#endif
    };
}

#include "swendsen_wang_selflabeling.hpp"

#undef XXX_NAMESPACE
