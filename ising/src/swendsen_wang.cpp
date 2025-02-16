#include <algorithm>
#include <cmath>
#include <omp.h>

#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    SwendsenWang_2D::SwendsenWang_2D()
    {
        const std::int32_t num_threads = omp_get_max_threads();
        rng.reserve(num_threads);
        for (std::int32_t i = 0; i < num_threads; ++i)
            rng.emplace_back(new LCG32(1 + i));
    }

    void SwendsenWang_2D::Update(Lattice<2>& lattice, const float temperature)
    {
        if (!cluster.Initialized())
            cluster.Resize(lattice.Extent());

        // Probability for adding aligned neighboring sites to the cluster.
        const float p_add = 1.0f - static_cast<float>(std::exp(-2.0f / temperature));

        // Labeling within tiles.
        AssignLabels(lattice, p_add);

        // Merging the tiles by establishing label equivalences.
        MergeLabels(lattice, p_add);

        // Resolve all labels for the final labeling.
        ResolveLabels();

        // Flip clusters.
        FlipClusters(lattice);
    }

    void SwendsenWang_2D::AssignLabels(Lattice<2>& lattice, const float p_add)
    {
        const std::int32_t n_0 = lattice.Extent()[0];
        const std::int32_t n_1 = lattice.Extent()[1];

        #pragma omp parallel for collapse(2)
        for (std::int32_t j = 0; j < n_1; j += chunk[1])
        {
            for (std::int32_t i = 0; i < n_0; i += chunk[0])
            {
                const std::array<int32_t, 2> n_offset{i, j};
                const std::array<int32_t, 2> n_sub{std::min(chunk[0], n_0 - i), std::min(chunk[1], n_1 - j)};
                if (n_sub[0] == chunk[0])
                {
                    // We can call a version of that method with the extent in 0-direction being
                    // a compile time constant (hopefully allowing the compiler to do better optimizations)
                    CCL_SelfLabeling<chunk[0]>(lattice, p_add, n_offset, n_sub);
                }
                else
                {
                    CCL_SelfLabeling(lattice, p_add, n_offset, n_sub);
                }
            }
        }
    }

    void SwendsenWang_2D::MergeLabels(Lattice<2>& lattice, const float p_add)
    {
        const std::int32_t n_0 = lattice.Extent()[0];
        const std::int32_t n_1 = lattice.Extent()[1];

        #pragma omp parallel
        {
            const std::uint32_t thread_id = omp_get_thread_num();
            constexpr std::int32_t buffer_size = chunk[0] + chunk[1];
            std::vector<float> buffer(buffer_size);

            #pragma omp for collapse(2)
            for (std::int32_t j = 0; j < n_1; j += chunk[1])
            {
                for (std::int32_t i = 0; i < n_0; i += chunk[0])
                {
                    rng[thread_id]->NextReal(buffer);

                    const std::int32_t jj_max = std::min(chunk[1], n_1 - j);
                    const std::int32_t ii_max = std::min(chunk[0], n_0 - i);

                    // Merge in 1-direction
                    for (std::int32_t ii = 0; ii < ii_max; ++ii)
                    {
                        if (buffer[ii] < p_add && lattice[j + jj_max - 1][i + ii] == lattice[(j + jj_max) % n_1][i + ii])
                        {
                            LabelType a = cluster[j + jj_max - 1][i + ii];
                            LabelType b = cluster[(j + jj_max) % n_1][i + ii];
                            if (a != b)
                                Merge(cluster.RawPointer(), a, b);
                        }
                    }

                    // merge in 0-direction
                    for (std::int32_t jj = 0; jj < jj_max; ++jj)
                    {
                        if (buffer[chunk[0] + jj] < p_add && lattice[j + jj][i + ii_max - 1] == lattice[j + jj][(i + ii_max) % n_0])
                        {
                            LabelType a = cluster[j + jj][i + ii_max - 1];
                            LabelType b = cluster[j + jj][(i + ii_max) % n_0];
                            if (a != b)
                                Merge(cluster.RawPointer(), a, b);
                        }
                    }
                }
            }
        }
    }

    // Helper method to establish label equivalences, thus merging clusters
    //
    // Important: the whole procedure works only because of 1) we start with all initial labels are
    // given by the 1-D index with ptr[X] = X (each site is its own cluster and root),
    // and 2) we always replace by the minimum when establishing the equivalence of two labels!
    //
    // Given an assumed label equivalence (here A and B are the same), this method alters the labels
    // pointed to by ptr such that afterwards ptr[A] and ptr[B] contain the min(A, B).
    // If in the mean time other threads have applied label equivalences, it might be possible that
    // any of ptr[A] and ptr[B] contains a value lower than min(A, B), e.g. if label C is equivalen to A
    // and C < A, then ptr[A] would hold the value C. In the mean time another thread could have
    // established ptr[C] = D, and so on.
    // However, in any case following these label equivalences down to the root (X = ptr[X]) gives the
    // final label for the clusters.
    //
    // The atomic_min() method makes sure that the field behind ptr is not corrupted when establishing
    // label equivalences in a multi-threaded context.
    void SwendsenWang_2D::Merge(LabelType* ptr, LabelType a, LabelType b)
    {
        // The loop is guaranteed to break somewhen (maybe not that obvious)
        while (true)
        {
            // c holds either the old value pointed to by ptr[b] in case of A is smaller,
            // or the actual minimum if the assumption that A is smaller is wrong
            LabelType c = AtomicMin(&ptr[b], a);

            // Successfully established the equivalence of A and B!
            // in the first loop iteration it might be that C != A, but if in the meantime no
            // other thread changes the equivalence, then ptr[B] = A and C will be equal to A
            if (c == a)
                break;

            // The assumption that A is smaller is true and we successfully established the
            // equivalence of A and B. We now have to adapt the already existing equivalence of
            // B and C (because previously ptr[B] held C). If C != B, we now have to establish
            // the equivalence of A and C, which is like calling this routine with B = C.
            if (c > a)
                b = c;

            // The assumption that A is smaller is false and C (!= B) is smaller than A.
            // we now have to establish the equivalence of A and C where not it is assumed that
            // C is the smaller one, which might have changed in the meantime.
            if (c < a)
            {
                b = a;
                a = c;
            }
        }
    }

    // Resolve all label equivalences
    void SwendsenWang_2D::ResolveLabels()
    {
        const std::size_t n_0 = cluster.Extent()[0];
        const std::size_t n_1 = cluster.Extent()[1];
        const std::size_t n = n_0 * n_1;
        auto* ptr = cluster.RawPointer();

        #pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i)
        {
            LabelType c = ptr[i];
            while (c != ptr[c])
                c = ptr[c];
            ptr[i] = c;
        }
    }

    // Clusters are flipped as a whole with probability 0.5.
    // As we use the 1-D index for the initial label assignment, the probability for the root label X
    // of each cluster to be either an even or an odd number is the same.
    // We thus flip a cluster only if X is odd, that is, if (X & 0x1) is equal to 0x1.
    // Flipping is implemented via a bitwise XOR operation.
    void SwendsenWang_2D::FlipClusters(Lattice<2>& lattice)
    {
        const std::size_t num_sites = lattice.NumSites();
        const auto* c_ptr = cluster.RawPointer();
        auto* ptr = lattice.RawPointer();

        #pragma omp parallel for simd
        for (std::size_t i = 0; i < num_sites; ++i)
            ptr[i] ^= (c_ptr[i] & 0x1);
    }
}
