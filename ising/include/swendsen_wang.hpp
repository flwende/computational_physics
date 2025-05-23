#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>

#include "array/multi_dimensional_array.hpp"
#include "thread_group/thread_group.hpp"
#include "atomic/atomic.hpp"
#include "simd/simd.hpp"
#include "random/random.hpp"
#include "lattice_mc.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    //using RNG = LCG32<DeviceName::CPU>;

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
        // Tile size for parallel processing: the innermost dimension should be a
        // multiple of the SIMD width of the target platform.
        static constexpr std::int32_t chunk[2] = {simd::Type<std::uint32_t>::width, 8};

        // Set to std::uint64_t if more then 4 billion labels.
        using LabelType = std::uint32_t;

        public:
            SwendsenWang_2D(ThreadGroup& thread_group);

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

            static constexpr std::int32_t Dimension = 2;

        protected:
            // Connected component labeling (ccl) based on an idea of Coddington and Baillie within tiles.
            template <std::int32_t N_0 = 0>
            void CCL_SelfLabeling(ThreadContext& context, Lattice<2>& lattice, const float p_add, const std::array<int32_t, 2>& n_offset, const std::array<int32_t, 2>& n_sub);

            // Loop over all tiles of the lattice and apply ccl_selflabeling.
            // Parameter p_add is the probability for adding aligned nearest
            // neighbor sites to a cluster.
            void AssignLabels(ThreadContext& context, Lattice<2>& lattice, const float p_add);

            // Connect all tiles.
            // Parameter p_add is the probability for adding aligned nearest
            // neighbor sites to the cluster.
            void MergeLabels(ThreadContext& context, Lattice<2>& lattice, const float p_add);

            // Helper method to establish label equivalences, thus merging clusters
            void Merge(LabelType* ptr, LabelType a, LabelType b);

            void Foo(ThreadContext&, Lattice<2>& lattice, const float) { /**/ }

            // Resolve all label equivalences.
            void ResolveLabels(ThreadContext& context);

            // Flip clusters.
            void FlipClusters(ThreadContext& context, Lattice<2>& lattice);

            // Multi-threading: reuse threads throughout MC updates.
            ThreadGroup& thread_group;

            // Cluster: the largest possible label is 0xFFFFFFFF
            MultiDimensionalArray<LabelType, 2> cluster;

            // Random number generator: if you use the lcg32 generator, make sure you are
            // compiling with RANDOM_SHUFFLE_STATE (otherwise, random numbers are too bad)
            std::vector<std::shared_ptr<RandomNumberGenerator>> rng;
    };
}

#include "swendsen_wang_selflabeling.hpp"

#undef XXX_NAMESPACE
