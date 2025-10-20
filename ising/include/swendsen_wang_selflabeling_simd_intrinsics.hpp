#pragma once

#include <limits>
#include <type_traits>

#include "atomic/atomic.hpp"
#include "misc/iota.hpp"
#include "simd/simd.hpp"
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
    template <std::uint32_t N_0>
    void SwendsenWang_2D<RNG, Target>::CCL_SelfLabeling(Context& context, Lattice<2>& lattice, const float p_add, const std::array<std::uint32_t, 2>& n_offset, const std::uint32_t n_1)
    {
        static_assert(N_0 == 4 || N_0 == 8 || N_0 == 16, "Only N_0=4,8,16 supported.");

        using namespace simd;

        auto& thread_group = static_cast<ThreadContext&>(context);
        const auto thread_id = thread_group.ThreadId();

        // Local copy of the tile.
        VariableLengthArray<std::uint32_t, 2> l(thread_group.StackMemory(), {N_0, n_1});
        // Local copy of the tile.
        VariableLengthArray<std::uint32_t, 2> c(thread_group.StackMemory(), {N_0, n_1});

        // Step 1 and 2.
        const auto ExcludeLastSpinInRow = MaskFromInteger<std::uint32_t>(N_0 == 16 ? 0x7FFF : (N_0 == 8 ? 0x7F : 0x7));
        auto vec_l_next = VecConvert<std::uint8_t, std::uint32_t>(VecLoad<std::uint8_t, 8 * N_0>(&lattice[n_offset[1] + 0][n_offset[0]]));
        for (std::uint32_t jj = 0; jj < n_1; ++jj)
        {
            const auto vec_l = vec_l_next;
            const auto vec_l_shifted = VecPermuteIdx<std::uint32_t>(vec_l, VecSet(Iota<std::uint32_t, N_0>(1, 15)));

            // Step 2: 0-direction -> set bit 1 if connected.
            const auto vec_random = VecLoad<float>(rng[thread_id]->NextRealArray());
            const auto mask_0 = MaskAnd<std::uint32_t>(MaskAnd<std::uint32_t>(
                VecCompareLT<float>(vec_random, VecSet1<float>(p_add)),
                VecCompareEQ<std::uint32_t>(vec_l, vec_l_shifted)),
                ExcludeLastSpinInRow);

            auto vec_l_encoding = VecOr_Masked<std::uint32_t>(vec_l, mask_0, vec_l, VecSet1<std::uint32_t>(0x2));

            if (jj < (n_1 - 1))
            {
                vec_l_next = VecConvert<std::uint8_t, std::uint32_t>(VecLoad<std::uint8_t, 8 * N_0>(&lattice[n_offset[1] + jj + 1][n_offset[0]]));

                // Step 2: 1-direction -> set bit 2 if connected.
                const auto vec_random = VecLoad<float>(rng[thread_id]->NextRealArray());
                const auto mask_1 = MaskAnd<std::uint32_t>(
                    VecCompareLT<float>(vec_random, VecSet1<float>(p_add)),
                    VecCompareEQ<std::uint32_t>(vec_l, vec_l_next));

                vec_l_encoding = VecOr_Masked<std::uint32_t>(vec_l_encoding, mask_1, vec_l_encoding, VecSet1<std::uint32_t>(0x4));
            }

            VecStore<std::uint32_t>(&l[jj][0], vec_l_encoding);
        }

        auto vec_c = VecSet(Iota<std::int32_t, N_0>(-std::int32_t{N_0}));
        for (std::uint32_t jj = 0; jj < n_1; ++jj)
        {
            vec_c = VecAdd<std::uint32_t>(vec_c, VecSet1<std::uint32_t>(N_0));
            VecStore<std::uint32_t>(&c[jj][0], vec_c);
        }

        // Step 4.
        auto break_loop = false;
        while (!break_loop)
        {
            break_loop = true;
            auto vec_a = VecLoad<std::uint32_t>(&c[0][0]);
            for (std::uint32_t jj = 0; jj < n_1; ++jj)
            {
                const auto vec_l = VecLoad<std::uint32_t>(&l[jj][0]);
                const auto mask_0 = VecCompareEQ<std::uint32_t>(VecAnd<std::uint32_t>(vec_l, VecSet1<std::uint32_t>(0x2)), VecSet1<std::uint32_t>(0x2));
                const auto mask_1 = MaskFromInteger<std::uint32_t>(MaskToInteger<std::uint32_t>(mask_0) << 1);
                const auto mask_2 = VecCompareEQ<std::uint32_t>(VecAnd<std::uint32_t>(vec_l, VecSet1<std::uint32_t>(0x4)), VecSet1<std::uint32_t>(0x4));

                //auto label_changes = true;
                while (true)
                {
                    auto vec_b = VecPermuteIdx<std::uint32_t>(vec_a, VecSet(Iota<std::uint32_t, N_0>(1, 15)));
                    const auto label_changes = MaskToInteger<std::uint32_t>(VecCompareNE_Masked<std::uint32_t>(mask_0, vec_a, vec_b));
                    if (!label_changes)
                        break;
                    else
                        break_loop = false;

                    vec_a = VecMin_Masked<std::uint32_t>(vec_a, mask_0, vec_a, vec_b);
                    vec_b = VecPermuteIdx<std::uint32_t>(vec_a, VecSet(Iota<std::uint32_t, N_0>(-1))); // The sequence will start with 0, 0, 1, 2,..
                    vec_a = VecMin_Masked<std::uint32_t>(vec_a, mask_1, vec_a, vec_b);
                }

                // No next row in 1-direction.
                if (jj < (n_1 - 1))
                {
                    auto vec_b = VecLoad<std::uint32_t>(&c[jj + 1][0]);
                    if (MaskToInteger<std::uint32_t>(VecCompareNE_Masked<std::uint32_t>(mask_2, vec_a, vec_b)))
                        break_loop = false;

                    VecStore<std::uint32_t>(&c[jj][0], VecMin_Masked<std::uint32_t>(vec_a, mask_2, vec_a, vec_b));
                    vec_a = VecMin_Masked<std::uint32_t>(vec_b, mask_2, vec_a, vec_b);
                    VecStore<std::uint32_t>(&c[jj + 1][0], vec_a);
                }
                else
                {
                    VecStore<std::uint32_t>(&c[jj][0], vec_a);
                }
            }
        }

        // Step 5: translate local to global labels.
        const auto n_0 = lattice.Extent()[0];
        constexpr auto div_shift = (N_0 == 16 ? 4 : (N_0 == 8 ? 3 : 2));
        for (std::uint32_t jj = 0; jj < n_1; ++jj)
        {
            auto vec_c = VecLoad<std::uint32_t>(&c[jj][0]);
            const auto vec_a = VecAnd<std::uint32_t>(vec_c, VecSet1<std::uint32_t>(N_0 - 1));
            const auto vec_b = VecShiftRight<std::uint32_t, div_shift>(vec_c);
            vec_c = VecAdd<std::uint32_t>(
                    VecMulLo<std::uint32_t>(VecAdd<std::uint32_t>(VecSet1<std::uint32_t>(n_offset[1]), vec_b), VecSet1<std::uint32_t>(n_0)),
                    VecAdd<std::uint32_t>(VecSet1<std::uint32_t>(n_offset[0]), vec_a));

            VecStore<std::uint32_t>(&cluster[n_offset[1] + jj][n_offset[0]], vec_c);
        }
    }
}

#undef XXX_NAMESPACE
