#pragma once

#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    // Connected component labeling (ccl) based on an idea of Coddington and Baillie within tiles
    //
    // References:
    //
    // * C. F. Baillie and P. D. Coddington,
    //            "Cluster identification algorithms for spin models - sequential and parallel",
    //            1991.
    //
    // * F. Wende and Th. Steinke,
    //            "Swendsen-Wang Multi-Cluster Algorithm for the 2D/3D Ising Model on Xeon Phi and GPU",
    //            SC'13 Proceedings, Article No. 83 ACM New York, NY, USA, 2013
    //
    //
    // General idea:
    //
    // 1. load tile into local memory: bit 0 is either set (spin up) or not (spin down).
    //
    // 2. for each site in the inner of the tile use bits 1 and 2 to encode whether its
    // neighboring site in 0- and 1-direction is aligned (has the same orientation) and whether
    // both of the two should be connected (depending on the value of p_add).
    //
    // 3. initialize the cluster associated with this tile so that all labels are unique (e.g. use the 1-D index).
    //
    // 4. go through all sites within the tile and for each one assign to it and its connected neighbor(s)
    // (see step 2) the minimum label. Iterate this step as long as labels change.
    //
    // 5. translate local labels to global labels.
    // Label L is mapped to L' = ((n_offset[1] + b) * n[0] + n_offset[0] + a, where
    // a = (L % n_0) and b = (L / n_0) and n_0 is either N_0 or n_sub[0].
    template <template <DeviceName> typename RNG, DeviceName Target>
    template <std::int32_t N_0>
    void SwendsenWang_2D<RNG, Target>::CCL_SelfLabeling(Context& context, Lattice<2>& lattice, const float p_add, const std::array<int32_t, 2>& n_offset, const std::array<int32_t, 2>& n_sub)
    {
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::uint32_t thread_id = thread_group.ThreadId();

        // Possible compiler optimization: N_0 has default value 0.
        // if the extent of the tile in 0-direction equals tile_size[0] (= multiple of the SIMD width),
        // the compiler can maybe apply some SIMD related optimizations
        const std::int32_t ii_max = (N_0 == 0 ? n_sub[0] : N_0);
        const std::int32_t jj_max = n_sub[1];
        // Local copy of the tile.
        //std::uint32_t l[jj_max][ii_max];
        // Local cluster.
        //std::uint32_t c[jj_max][ii_max];
        // Temporaries.
        //std::uint32_t tmp[ii_max];
        auto i_ptr = reinterpret_cast<std::uint32_t*>(alloca((2 * (ii_max * jj_max) + ii_max) * sizeof(std::uint32_t)));
        NonOwningMultiDimensionalArray<std::uint32_t, 2> l{i_ptr, std::array<std::int32_t, 2>{ii_max, jj_max}};
        NonOwningMultiDimensionalArray<std::uint32_t, 2> c{i_ptr + ii_max * jj_max, std::array<std::int32_t, 2>{ii_max, jj_max}};
        NonOwningMultiDimensionalArray<std::uint32_t, 1> tmp{i_ptr + 2 * ii_max * jj_max, std::array<std::int32_t, 1>{ii_max}};

        // Random numbers.
        std::vector<float> buffer(tile_size[0]);

        // Step 1.
        for (std::int32_t jj = 0; jj < jj_max; ++jj)
        {
            #pragma omp simd
            for (std::int32_t ii = 0; ii < ii_max; ++ii)
                l[jj][ii] = lattice[n_offset[1] + jj][n_offset[0] + ii];
        }

        // Step 2: 0-direction -> set bit 1 if connected.
        for (std::int32_t jj = 0; jj < jj_max; ++jj)
        {
            rng[thread_id]->NextReal(buffer);

            for (std::int32_t ii = 0; ii < (ii_max - 1); ++ii)
                tmp[ii] = l[jj][ii + 1];
            tmp[ii_max - 1] = 0x2;

            #pragma omp simd
            for (std::int32_t ii = 0; ii < ii_max; ++ii)
            {
                std::uint32_t l_0 = l[jj][ii];
                if (l_0 == tmp[ii] && buffer[ii] < p_add)
                    l_0 |= 0x2;
                l[jj][ii] = l_0;
            }
        }

        // Step 2: 1-direction -> set bit 2 if connected.
        for (std::int32_t jj = 0; jj < (jj_max - 1); ++jj)
        {
            rng[thread_id]->NextReal(buffer);

            #pragma omp simd
            for (std::int32_t ii = 0; ii < ii_max; ++ii)
            {
                std::uint32_t l_0 = l[jj][ii];
                if ((l_0 & 0x1) == (l[jj + 1][ii] & 0x1) && buffer[ii] < p_add)
                    l_0 |= 0x4;
                l[jj][ii] = l_0;
            }
        }

        // Step 3: use 1-D index for the initial labeling (unique).
        for (std::int32_t jj = 0; jj < jj_max; ++jj)
        {
            #pragma omp simd
            for (std::int32_t ii = 0; ii < ii_max; ++ii)
                c[jj][ii] = jj * ii_max + ii;
        }

        // Step 4.
        bool break_loop = false;
        while (!break_loop)
        {
            break_loop = true;
            for (std::int32_t jj = 0; jj < jj_max; ++jj)
            {
                bool label_changes = true;
                while (label_changes)
                {
                    #pragma omp simd
                    for (std::int32_t ii = 0; ii < ii_max; ++ii)
                        tmp[ii] = (l[jj][ii] & 0x2);

                    label_changes = false;
                    for (std::int32_t ii = 0; ii < (ii_max - 1); ++ii)
                    {
                        if (tmp[ii])
                        {
                            const std::uint32_t a = c[jj][ii];
                            const std::uint32_t b = c[jj][ii + 1];
                            if (a != b)
                            {
                                // Replace both labels by their minimum.
                                const std::uint32_t ab = std::min(a, b);
                                c[jj][ii] = ab;
                                c[jj][ii + 1] = ab;
                                label_changes = true;
                            }
                        }
                    }

                    if (label_changes)
                        break_loop = false;
                }

                // No next row in 1-direction.
                if (jj == (jj_max - 1))
                    continue;

                #pragma omp simd
                for (std::int32_t ii = 0; ii < ii_max; ++ii)
                    tmp[ii] = (l[jj][ii] & 0x4);

                std::uint32_t counter = 0;
                #pragma omp simd reduction(+ : counter)
                for (std::int32_t ii = 0; ii < ii_max; ++ii)
                {
                    if (tmp[ii])
                    {
                        const std::uint32_t a = c[jj][ii];
                        const std::uint32_t b = c[jj + 1][ii];
                        if (a != b)
                        {
                            // Replace both labels by their minimum
                            const std::uint32_t ab = std::min(a, b);
                            c[jj][ii] = ab;
                            c[jj + 1][ii] = ab;
                            counter++;
                        }
                    }
                }

                if (counter)
                    break_loop = false;
            }
        }

        // Step 5: translate local to global labels.
        const std::int32_t n_0 = lattice.Extent()[0];
        for (std::int32_t jj = 0; jj < jj_max; ++jj)
        {
            for (std::int32_t ii = 0; ii < ii_max; ++ii)
            {
                const std::int32_t a = c[jj][ii] % ii_max;
                const std::int32_t b = c[jj][ii] / ii_max;
                cluster[n_offset[1] + jj][n_offset[0] + ii] = static_cast<LabelType>(n_offset[1] + b) * n_0 + (n_offset[0] + a);
            }
        }
    }
}

#undef XXX_NAMESPACE
