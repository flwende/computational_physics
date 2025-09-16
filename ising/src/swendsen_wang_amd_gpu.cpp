#if defined __HIPCC__

#include <algorithm>
#include <cmath>
#include <iostream>

#include "environment/environment.hpp"
#include "swendsen_wang.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

// Dynamic shared memory for GPU kernels.
extern __shared__ std::byte shared_memory[];

namespace XXX_NAMESPACE
{
    using GpuLabelType = std::uint16_t;

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::InitializeGpuRngState()
    {
        if (!gpu_rng_state.get())
        {
            std::cout << "Initializing GPU RNG state .. ";

            RngState* ptr{};
            SafeCall(hipSetDevice(target.DeviceId()));
            SafeCall(hipMalloc(&ptr, rng_state.size() * sizeof(RngState)));
            SafeCall(hipMemcpy(ptr, rng_state.data(), rng_state.size() * sizeof(RngState), hipMemcpyHostToDevice));
            gpu_rng_state.reset(ptr);

            std::cout << "Done." << std::endl;
        }
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::InitializeGpuCluster(const LabelType num_sites)
    {
        if (!gpu_cluster.get())
        {
            std::cout << "Initializing GPU cluster .. ";

            LabelType* ptr{};
            SafeCall(hipSetDevice(target.DeviceId()));
            SafeCall(hipMalloc(&ptr, num_sites * sizeof(LabelType)));
            gpu_cluster.reset(ptr);

            std::cout << "Done." << std::endl;
        }
    }

    template <std::int32_t N>
    __host__ __device__
    std::int32_t Ceil(const std::int32_t value)
    {
        return ((value + N - 1) / N) * N;
    }

    template <typename RngState>
    __device__
    RngState* LoadRngState(const RngState* rng_state)
    {
        RngState* shared_rng_state = reinterpret_cast<RngState*>(shared_memory);
        shared_rng_state->Load(rng_state[blockIdx.x]);
        return shared_rng_state;
    }

    template <typename RngState>
    __device__
    void UnloadRngState(const RngState* shared_rng_state, RngState* rng_state)
    {
        shared_rng_state->Unload(rng_state[blockIdx.x]);
    }

    template <typename Spin, typename LabelType, typename RngState>
    __global__
    void AssignLabels_Kernel(const Spin* lattice, const std::int32_t extent_0, const std::int32_t extent_1,
        const std::int32_t n_sub_0, const std::int32_t n_sub_1,
        LabelType* cluster, RngState* rng_state, const float p_add)
    {
        const std::int32_t n_0 = extent_0 / n_sub_0;
        const std::int32_t n_1 = extent_1 / n_sub_1;
        const std::int32_t n_total = n_0 * n_1;

        RngState* state = LoadRngState(rng_state);
        Spin* sub_lattice = reinterpret_cast<Spin*>(shared_memory + Ceil<64>(sizeof(RngState)));
        GpuLabelType* sub_cluster = reinterpret_cast<GpuLabelType*>(shared_memory + Ceil<64>(sizeof(RngState)) +
            n_sub_0 * n_sub_1 * sizeof(Spin));
        
        for (std::int32_t n = blockIdx.x; n < n_total; n += gridDim.x)
        {
            const std::int32_t bx = (n % n_0);
            const std::int32_t by = (n / n_0);

            // Load the sub-lattice for the current block.
            sub_lattice[threadIdx.y * n_sub_0 + threadIdx.x] = lattice[(by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + threadIdx.x];
            sub_lattice[threadIdx.y * n_sub_0 + blockDim.x + threadIdx.x] = lattice[(by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + blockDim.x + threadIdx.x];
            
            // Initialize the cluster labels.
            sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x] = threadIdx.y * n_sub_0 + threadIdx.x;
            sub_cluster[threadIdx.y * n_sub_0 + blockDim.x + threadIdx.x] = threadIdx.y * n_sub_0 + blockDim.x + threadIdx.x;

            // Connect 1-direction.
            {
                state->Update();
                const float r_1 = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
                if (sub_lattice[threadIdx.y * n_sub_0 + 2 * threadIdx.x] == sub_lattice[threadIdx.y * n_sub_0 + 2 * threadIdx.x + 1] &&
                    r_1 < p_add)
                {
                    sub_lattice[threadIdx.y * n_sub_0 + 2 * threadIdx.x] |= 0x2;
                }

                state->Update();
                const float r_2 = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
                if (threadIdx.x < (blockDim.x - 1) &&
                    sub_lattice[threadIdx.y * n_sub_0 + 2 * threadIdx.x + 1] == sub_lattice[threadIdx.y * n_sub_0 + 2 * threadIdx.x + 2] &&
                    r_2 < p_add)
                {
                    sub_lattice[threadIdx.y * n_sub_0 + 2 * threadIdx.x + 1] |= 0x2;
                }
            }

            // Connect 2-direction.
            if (threadIdx.y < (blockDim.y - 1))
            {
                state->Update();
                const float r_1 = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
                if (sub_lattice[threadIdx.y * n_sub_0 + threadIdx.x] == sub_lattice[(threadIdx.y + 1) * n_sub_0 + threadIdx.x] &&
                    r_1 < p_add)
                {
                    sub_lattice[threadIdx.y * n_sub_0 + threadIdx.x] |= 0x4;
                }

                state->Update();
                const float r_2 = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
                if (sub_lattice[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x] == sub_lattice[(threadIdx.y + 1) * n_sub_0 + blockIdx.x + threadIdx.x] &&
                    r_2 < p_add)
                {
                    sub_lattice[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x] |= 0x4;
                }
            }

            #if defined RANDOM_SHUFFLE_STATE
            state->Shuffle();
            #endif

            // Reduce labels.
            bool labels_changed = true;
            while (__any(labels_changed))
            {
                labels_changed = false;

                // 1-drection reduction.
                if (sub_lattice[threadIdx.y * n_sub_0 + threadIdx.x] & 0x2)
                {
                    const auto a = sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x];
                    const auto b = sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x + 1];
                    if (a != b)
                    {
                        const auto ab = a < b ? a : b;
                        sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x] = ab;
                        sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x + 1] = ab;
                        labels_changed = true;
                    }
                }

                if (sub_lattice[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x] & 0x2)
                {
                    const auto a = sub_cluster[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x];
                    const auto b = sub_cluster[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x + 1];
                    if (a != b)
                    {
                        const auto ab = a < b ? a : b;
                        sub_cluster[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x] = ab;
                        sub_cluster[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x + 1] = ab;
                        labels_changed = true;
                    }
                }

                // 2-drection reduction.
                if (sub_lattice[threadIdx.y * n_sub_0 + threadIdx.x] & 0x4)
                {
                    const auto a = sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x];
                    const auto b = sub_cluster[(threadIdx.y + 1) * n_sub_0 + threadIdx.x];
                    if (a != b)
                    {
                        const auto ab = a < b ? a : b;
                        sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x] = ab;
                        sub_cluster[(threadIdx.y + 1) * n_sub_0 + threadIdx.x] = ab;
                        labels_changed = true;
                    }
                }

                if (sub_lattice[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x] & 0x4)
                {
                    const auto a = sub_cluster[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x];
                    const auto b = sub_cluster[(threadIdx.y + 1) * n_sub_0 + blockIdx.x + threadIdx.x];
                    if (a != b)
                    {
                        const auto ab = a < b ? a : b;
                        sub_cluster[threadIdx.y * n_sub_0 + blockIdx.x + threadIdx.x] = ab;
                        sub_cluster[(threadIdx.y + 1) * n_sub_0 + blockIdx.x + threadIdx.x] = ab;
                        labels_changed = true;
                    }
                }
            }

            // Write back cluster.
            {
                const std::int32_t a = sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x] % n_sub_0;
                const std::int32_t b = sub_cluster[threadIdx.y * n_sub_0 + threadIdx.x] / n_sub_0;
                cluster[(by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + threadIdx.x] = (by * n_sub_1 + b) * extent_0 + bx * n_sub_0 + a;
            }
            {
                const std::int32_t a = sub_cluster[threadIdx.y * n_sub_0 + blockDim.x + threadIdx.x] % n_sub_0;
                const std::int32_t b = sub_cluster[threadIdx.y * n_sub_0 + blockDim.x + threadIdx.x] / n_sub_0;
                cluster[(by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + blockDim.x + threadIdx.x] = (by * n_sub_1 + b) * extent_0 + bx * n_sub_0 + a;
            }
        }   

        UnloadRngState(state, rng_state);
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::AssignLabels(Context& context, Lattice<2>& lattice, const float p_add)
    {
        static bool ran_already = false;
        //if (ran_already)
        //    return;

        HipContext& gpu = static_cast<HipContext&>(context);
        const std::uint32_t num_thread_blocks = 16;//gpu.Device().Concurrency();

        //std::cout << "Number of thread blocks: " << num_thread_blocks << std::endl;

        const std::int32_t extent_0 = lattice.Extent()[0];
        const std::int32_t extent_1 = lattice.Extent()[1];

        // Configure kernel launch.
        const dim3 grid{num_thread_blocks, 1, 1};
        const dim3 block{static_cast<std::uint32_t>(tile_size[0]), static_cast<std::uint32_t>(tile_size[1]), 1};
        const std::size_t shared_mem_bytes = Ceil<64>(sizeof(RngState)) +
            2 * tile_size[0] * tile_size[1] * (sizeof(Lattice<2>::Spin) + sizeof(GpuLabelType));

        //std::cout << "Shmem: " << shared_mem_bytes << " Bytes" << std::endl;
        SafeCall(hipSetDevice(gpu.Id()));
        AssignLabels_Kernel<<<grid, block, shared_mem_bytes>>>(lattice.RawGpuPointer(),
            extent_0, extent_1, 2 * tile_size[0], tile_size[1],
            gpu_cluster.get(), gpu_rng_state.get(), p_add);

        gpu.Synchronize();
        
        if (false && !ran_already)
        {
            SafeCall(hipMemcpy(cluster.RawPointer(), gpu_cluster.get(), lattice.NumSites() * sizeof(LabelType), hipMemcpyDeviceToHost));

            for (std::int32_t j = 0; j < extent_1; ++j)
            {
                for (std::int32_t i = 0; i < extent_0; ++i)
                {
                    std::cout << std::setw(3) << cluster[j][i] << " ";
                }
                std::cout << std::endl;
            }

            ran_already = true;
        }
        /*
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::int32_t thread_id = thread_group.ThreadId();
        const std::int32_t num_threads = thread_group.NumThreads();

        const std::int32_t extent_0 = lattice.Extent()[0];
        const std::int32_t extent_1 = lattice.Extent()[1];

        const std::int32_t n_0 = extent_0 / tile_size[0];
        const std::int32_t n_1 = extent_1 / tile_size[1];
        const std::int32_t n_total = n_0 * n_1;

        const std::int32_t n_chunk = (n_total + num_threads - 1) / num_threads;
        const std::int32_t n_start = thread_id * n_chunk;
        const std::int32_t n_end = std::min(n_start + n_chunk, n_total);

        for (std::int32_t n = n_start; n < n_end; ++n)
        {
            const std::int32_t j = (n / n_0) * tile_size[1];
            const std::int32_t i = (n % n_0) * tile_size[0];
            const std::array<int32_t, 2> n_offset{i, j};
            const std::array<int32_t, 2> n_sub{std::min(tile_size[0], extent_0 - i), std::min(tile_size[1], extent_1 - j)};
            if (n_sub[0] == tile_size[0])
            {
                // We can call a version of that method with the extent in 0-direction being
                // a compile time constant (hopefully allowing the compiler to do better optimizations)
                CCL_SelfLabeling<tile_size[0]>(context, lattice, p_add, n_offset, n_sub);
            }
            else
            {
                CCL_SelfLabeling(context, lattice, p_add, n_offset, n_sub);
            }
        }
        */
    }

    template <typename LabelType>
    __device__
    void Merge(LabelType* ptr, LabelType a, LabelType b);

    template <typename Spin, typename LabelType, typename RngState>
    __global__
    void MergeLabels_Kernel(const Spin* lattice, const std::int32_t extent_0, const std::int32_t extent_1,
        const std::int32_t n_sub_0, const std::int32_t n_sub_1,
        LabelType* cluster, RngState* rng_state, const float p_add)
    {
        const std::int32_t n_0 = extent_0 / n_sub_0;
        const std::int32_t n_1 = extent_1 / n_sub_1;
        const std::int32_t n_total = n_0 * n_1;

        RngState* state = LoadRngState(rng_state);

        for (std::int32_t n = blockIdx.x; n < n_total; n += gridDim.x)
        {
            const std::int32_t bx = (n % n_0);
            const std::int32_t by = (n / n_0);

            state->Update();
            const float r_1 = 2.3283064370807974e-10f * state->Get(threadIdx.x);
            state->Update();
            const float r_2 = 2.3283064370807974e-10f * state->Get(threadIdx.x);

            if (threadIdx.x < n_sub_1)
            {
                const std::int32_t idx = ((by * n_sub_1) + threadIdx.x) * extent_0 + (bx + 1) * n_sub_0 - 1;
                const std::int32_t idx_p1 = idx + 1 - (bx == (n_0 - 1) ? extent_0 : 0);

                if (lattice[idx] == lattice[idx_p1] && r_1 < p_add)
                {
                    LabelType a = cluster[idx];
                    LabelType b = cluster[idx_p1];
                    if (a != b)
                        Merge(cluster, a, b);
                }
            }

            if (threadIdx.x < n_sub_0)
            {
                const std::int32_t idx = (((by + 1) * n_sub_1) - 1) * extent_0 + bx * n_sub_0 + threadIdx.x;
                const std::int32_t idx_p2 = idx + extent_0 - (by == (n_1 - 1) ? extent_0 * extent_1 : 0);

                if (lattice[idx] == lattice[idx_p2] && r_2 < p_add)
                {
                    LabelType a = cluster[idx];
                    LabelType b = cluster[idx_p2];
                    if (a != b)
                        Merge(cluster, a, b);
                }
            }
        }

        UnloadRngState(state, rng_state);
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::MergeLabels(Context& context, Lattice<2>& lattice, const float p_add)
    {
        static bool ran_already = true;
        //if (ran_already)
        //    return;

        HipContext& gpu = static_cast<HipContext&>(context);
        const std::uint32_t num_thread_blocks = gpu.Device().Concurrency();
        //const std::uint32_t num_thread_blocks = 16;

        //std::cout << "Number of thread blocks: " << num_thread_blocks << std::endl;

        const std::int32_t extent_0 = lattice.Extent()[0];
        const std::int32_t extent_1 = lattice.Extent()[1];

        // Configure kernel launch.
        const dim3 grid{num_thread_blocks, 1, 1};
        const dim3 block{static_cast<std::uint32_t>(std::max(2 * tile_size[0], tile_size[1])), 1, 1};
        const std::size_t shared_mem_bytes = Ceil<64>(sizeof(RngState));

        //std::cout << "Shmem: " << shared_mem_bytes << " Bytes" << std::endl;
        
        SafeCall(hipSetDevice(gpu.Id()));
        MergeLabels_Kernel<<<grid, block, shared_mem_bytes>>>(lattice.RawGpuPointer(),
            extent_0, extent_1, 2 * tile_size[0], tile_size[1],
            gpu_cluster.get(), gpu_rng_state.get(), p_add);

        gpu.Synchronize();
        
        if (false && !ran_already)
        {
            SafeCall(hipMemcpy(cluster.RawPointer(), gpu_cluster.get(), lattice.NumSites() * sizeof(LabelType), hipMemcpyDeviceToHost));

            for (std::int32_t j = 0; j < extent_1; ++j)
            {
                for (std::int32_t i = 0; i < extent_0; ++i)
                {
                    std::cout << std::setw(3) << cluster[j][i] << " ";
                }
                std::cout << std::endl;
            }

            ran_already = true;
        }

        /*
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::int32_t thread_id = thread_group.ThreadId();
        const std::int32_t num_threads = thread_group.NumThreads();

        const std::int32_t extent_0 = lattice.Extent()[0];
        const std::int32_t extent_1 = lattice.Extent()[1];

        const std::int32_t n_0 = extent_0 / tile_size[0];
        const std::int32_t n_1 = extent_1 / tile_size[1];
        const std::int32_t n_total = n_0 * n_1;

        const std::int32_t n_chunk = (n_total + num_threads - 1) / num_threads;
        const std::int32_t n_start = thread_id * n_chunk;
        const std::int32_t n_end = std::min(n_start + n_chunk, n_total);

        constexpr std::int32_t buffer_size = tile_size[0] + tile_size[1];
        std::vector<float> buffer(buffer_size);

        for (std::int32_t n = n_start; n < n_end; ++n)
        {
            const std::int32_t j = (n / n_0) * tile_size[1];
            const std::int32_t i = (n % n_0) * tile_size[0];

            rng[thread_id]->NextReal(buffer);

            const std::int32_t jj_max = std::min(tile_size[1], extent_1 - j);
            const std::int32_t ii_max = std::min(tile_size[0], extent_0 - i);

            // Merge in 1-direction
            for (std::int32_t ii = 0; ii < ii_max; ++ii)
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
            for (std::int32_t jj = 0; jj < jj_max; ++jj)
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
        */
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
    template <typename LabelType>
    __device__
    void Merge(LabelType* ptr, LabelType a, LabelType b)
    {
        // The loop is guaranteed to break somewhen (maybe not that obvious)
        while (true)
        {
            // c holds either the old value pointed to by ptr[b] in case of A is smaller,
            // or the actual minimum if the assumption that A is smaller is wrong
            LabelType c = atomicMin(ptr + b, a);

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

    template <typename LabelType>
    __global__
    void ResolveLabels_Kernel(LabelType* cluster, const std::int32_t extent_0, const std::int32_t extent_1,
        const std::int32_t n_sub_0, const std::int32_t n_sub_1)
    {
        const std::int32_t n_0 = extent_0 / n_sub_0;
        const std::int32_t n_1 = extent_1 / n_sub_1;
        const std::int32_t n_total = n_0 * n_1;

        for (std::int32_t n = blockIdx.x; n < n_total; n += gridDim.x)
        {
            const std::int32_t bx = (n % n_0);
            const std::int32_t by = (n / n_0);
            const std::int32_t idx = (by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + threadIdx.x;
            /*
            LabelType c = cluster[idx];
            while (c != cluster[c])
                c = cluster[c];
            */
            //cluster[idx] = c;
            //atomicExch(cluster + idx, c);
            
            
            //LabelType c = cluster[idx];
            LabelType c = atomicOr(cluster + idx, 0x0);
            LabelType a{};
            do
            {
                a = c;
                //c = cluster[c];
                c = atomicOr(cluster + c, 0x0);
            }
            while (a != c);
            atomicExch(cluster + idx, c);
        }
    }


    // Resolve all label equivalences
    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::ResolveLabels(Context& context)
    {
        static bool ran_already = false;
        static std::int32_t iteration = 0;
        //if (ran_already)
        //    return;

        HipContext& gpu = static_cast<HipContext&>(context);
        const std::uint32_t num_thread_blocks = gpu.Device().Concurrency();
        //const std::uint32_t num_thread_blocks = 16;

        //std::cout << "Number of thread blocks: " << num_thread_blocks << std::endl;

        const std::int32_t extent_0 = cluster.Extent()[0];
        const std::int32_t extent_1 = cluster.Extent()[1];

        // Configure kernel launch.
        const dim3 grid{num_thread_blocks, 1, 1};
        const dim3 block{static_cast<std::uint32_t>(2 * tile_size[0]), static_cast<std::uint32_t>(tile_size[1]), 1};
        
        SafeCall(hipSetDevice(gpu.Id()));
        ResolveLabels_Kernel<<<grid, block>>>(gpu_cluster.get(), extent_0, extent_1, 2 * tile_size[0], tile_size[1]);

        gpu.Synchronize();
        
        //if (!ran_already)
        if (++iteration == 100)
        {
            SafeCall(hipMemcpy(cluster.RawPointer(), gpu_cluster.get(), extent_0 * extent_1 * sizeof(LabelType), hipMemcpyDeviceToHost));

            for (std::int32_t j = 0; j < extent_1; ++j)
            {
                for (std::int32_t i = 0; i < extent_0; ++i)
                {
                    std::cout << std::setw(3) << cluster[j][i] << " ";
                }
                std::cout << std::endl;
            }

            ran_already = true;
        }

        /*
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::int32_t threads_id = thread_group.ThreadId();
        const std::int32_t num_threads = thread_group.NumThreads();

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
        */
    }

    template <typename Spin, typename LabelType>
    __global__
    void FlipClusters_Kernel(Spin* lattice, const std::int32_t extent_0, const std::int32_t extent_1,
        const std::int32_t n_sub_0, const std::int32_t n_sub_1,
        const LabelType* cluster)
    {
        const std::int32_t n_0 = extent_0 / n_sub_0;
        const std::int32_t n_1 = extent_1 / n_sub_1;
        const std::int32_t n_total = n_0 * n_1;

        for (std::int32_t n = blockIdx.x; n < n_total; n += gridDim.x)
        {
            const std::int32_t bx = (n % n_0);
            const std::int32_t by = (n / n_0);

            lattice[(by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + threadIdx.x] ^=
                (cluster[(by * n_sub_1 + threadIdx.y) * extent_0 + bx * n_sub_0 + threadIdx.x] & 0x1);
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
        HipContext& gpu = static_cast<HipContext&>(context);
        const std::uint32_t num_thread_blocks = gpu.Device().Concurrency();
        //const std::uint32_t num_thread_blocks = 16;

        //std::cout << "Number of thread blocks: " << num_thread_blocks << std::endl;

        const std::int32_t extent_0 = cluster.Extent()[0];
        const std::int32_t extent_1 = cluster.Extent()[1];

        // Configure kernel launch.
        const dim3 grid{num_thread_blocks, 1, 1};
        const dim3 block{static_cast<std::uint32_t>(2 * tile_size[0]), static_cast<std::uint32_t>(tile_size[1]), 1};
        
        SafeCall(hipSetDevice(gpu.Id()));
        FlipClusters_Kernel<<<grid, block>>>(lattice.RawGpuPointer(),
            extent_0, extent_1, 2 * tile_size[0], tile_size[1],
            gpu_cluster.get());

        gpu.Synchronize();
        
        /*
        auto& thread_group = static_cast<ThreadContext&>(context);
        const std::int32_t threads_id = thread_group.ThreadId();
        const std::int32_t num_threads = thread_group.NumThreads();

        const std::size_t num_sites = lattice.NumSites();
        const std::size_t chunk_size = (num_sites + num_threads - 1) / num_threads;
        const std::size_t start = threads_id * chunk_size;
        const std::size_t end = std::min(start + chunk_size, num_sites);

        const auto* c_ptr = cluster.RawPointer();
        auto* ptr = lattice.RawPointer();

        #pragma omp simd
        for (std::size_t i = start; i < end; ++i)
            ptr[i] ^= (c_ptr[i] & 0x1);
        */
    }

    // Explicit template instantiation.
    template class SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>;
}

#endif