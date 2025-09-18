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
    void SwendsenWang_2D<RNG, Target>::AssignLabels(Context& context, Lattice<2>& lattice, const float p_add)
    {
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::uint32_t thread_id = thread_group.ThreadId();
        const std::uint32_t num_threads = thread_group.NumThreads();

        const std::uint32_t extent_0 = lattice.Extent()[0];
        const std::uint32_t extent_1 = lattice.Extent()[1];

        const std::uint32_t n_0 = extent_0 / tile_size[0];
        const std::uint32_t n_1 = extent_1 / tile_size[1];
        const std::uint32_t n_total = n_0 * n_1;

        const std::uint32_t n_chunk = (n_total + num_threads - 1) / num_threads;
        const std::uint32_t n_start = thread_id * n_chunk;
        const std::uint32_t n_end = std::min(n_start + n_chunk, n_total);

        for (std::uint32_t n = n_start; n < n_end; ++n)
        {
            const std::uint32_t j = (n / n_0) * tile_size[1];
            const std::uint32_t i = (n % n_0) * tile_size[0];
            const std::array<uint32_t, 2> n_offset{i, j};
            const std::array<uint32_t, 2> n_sub{std::min(tile_size[0], extent_0 - i), std::min(tile_size[1], extent_1 - j)};
            if (n_sub[0] == WavefrontSize)
            {
                // We can call a version of that method with the extent in 0-direction being
                // a compile time constant (hopefully allowing the compiler to do better optimizations)
                CCL_SelfLabeling<WavefrontSize>(context, lattice, p_add, n_offset, n_sub);
            }
            else
            {
                CCL_SelfLabeling(context, lattice, p_add, n_offset, n_sub);
            }
        }
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::MergeLabels(Context& context, Lattice<2>& lattice, const float p_add)
    {
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::uint32_t thread_id = thread_group.ThreadId();
        const std::uint32_t num_threads = thread_group.NumThreads();

        const std::uint32_t extent_0 = lattice.Extent()[0];
        const std::uint32_t extent_1 = lattice.Extent()[1];

        const std::uint32_t n_0 = extent_0 / tile_size[0];
        const std::uint32_t n_1 = extent_1 / tile_size[1];
        const std::uint32_t n_total = n_0 * n_1;

        const std::uint32_t n_chunk = (n_total + num_threads - 1) / num_threads;
        const std::uint32_t n_start = thread_id * n_chunk;
        const std::uint32_t n_end = std::min(n_start + n_chunk, n_total);

        const std::uint32_t buffer_size = tile_size[0] + tile_size[1];
        std::vector<float> buffer(buffer_size);

        for (std::uint32_t n = n_start; n < n_end; ++n)
        {
            const std::uint32_t j = (n / n_0) * tile_size[1];
            const std::uint32_t i = (n % n_0) * tile_size[0];

            rng[thread_id]->NextReal(buffer);

            const std::uint32_t jj_max = std::min(tile_size[1], extent_1 - j);
            const std::uint32_t ii_max = std::min(tile_size[0], extent_0 - i);

            // Merge in 1-direction
            for (std::uint32_t ii = 0; ii < ii_max; ++ii)
            {
                if (buffer[ii] < p_add && lattice[j + jj_max - 1][i + ii] == lattice[(j + jj_max) % extent_1][i + ii])
                {
                    LabelType a = cluster[j + jj_max - 1][i + ii];
                    LabelType b = cluster[(j + jj_max) % extent_1][i + ii];
                    if (a != b)
                        Merge(cluster.RawPointer(), a, b);
                }
            }

            // merge in 0-direction
            for (std::uint32_t jj = 0; jj < jj_max; ++jj)
            {
                if (buffer[tile_size[0] + jj] < p_add && lattice[j + jj][i + ii_max - 1] == lattice[j + jj][(i + ii_max) % extent_0])
                {
                    LabelType a = cluster[j + jj][i + ii_max - 1];
                    LabelType b = cluster[j + jj][(i + ii_max) % extent_0];
                    if (a != b)
                        Merge(cluster.RawPointer(), a, b);
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
    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::Merge(LabelType* ptr, LabelType a, LabelType b)
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
    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::ResolveLabels(Context& context)
    {
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::uint32_t threads_id = thread_group.ThreadId();
        const std::uint32_t num_threads = thread_group.NumThreads();

        const std::size_t num_sites = cluster.Extent()[0] * cluster.Extent()[1];
        const std::size_t chunk_size = (num_sites + num_threads - 1) / num_threads;
        const std::size_t start = threads_id * chunk_size;
        const std::size_t end = std::min(start + chunk_size, num_sites);

        auto* ptr = cluster.RawPointer();

        for (std::size_t i = start; i < end; ++i)
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
    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::FlipClusters(Context& context, Lattice<2>& lattice)
    {
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::uint32_t threads_id = thread_group.ThreadId();
        const std::uint32_t num_threads = thread_group.NumThreads();

        const std::size_t num_sites = lattice.NumSites();
        const std::size_t chunk_size = (num_sites + num_threads - 1) / num_threads;
        const std::size_t start = threads_id * chunk_size;
        const std::size_t end = std::min(start + chunk_size, num_sites);

        const auto* c_ptr = cluster.RawPointer();
        auto* ptr = lattice.RawPointer();

        #pragma omp simd
        for (std::size_t i = start; i < end; ++i)
            ptr[i] ^= (c_ptr[i] & 0x1);
    }

    // Explicit template instantiation.
    template class SwendsenWang_2D<LCG32, DeviceName::CPU>;
}