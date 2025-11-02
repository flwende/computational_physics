#if defined(__HIPCC__)

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
    using GpuSpin = std::uint16_t;
    using GpuLabelType = std::uint16_t;

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::InitializeGpuRngState()
    {
        if (!gpu_rng_state.get())
        {
            std::cout << "Initializing GPU RNG state .. ";

            RngState* ptr {};
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

            LabelType* ptr {};
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
    RngState* LoadRngState(const RngState* rng_state, const std::uint32_t rng_id)
    {
        RngState* shared_rng_state = reinterpret_cast<RngState*>(shared_memory);
        shared_rng_state->Load(rng_state[rng_id]);
        return shared_rng_state;
    }

    template <typename RngState>
    __device__
    void UnloadRngState(const RngState* shared_rng_state, RngState* rng_state, const std::uint32_t rng_id)
    {
        shared_rng_state->Unload(rng_state[rng_id]);
    }

    template <typename Spin, typename LabelType, typename RngState>
    __global__
    void AssignLabels_Kernel(const Spin* lattice, const std::uint32_t extent_0, const std::uint32_t extent_1,
        LabelType* cluster, RngState* rng_state, const std::uint32_t rng_state_shift, const float p_add)
    {
        const auto rng_id = (blockIdx.y * gridDim.x + blockIdx.x + rng_state_shift) % (gridDim.x * gridDim.y);
        auto* state = LoadRngState(rng_state, rng_id);

        auto r = float4{};
        state->Update();
        r.x = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
        state->Update();
        r.y = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
        state->Update();
        r.z = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
        state->Update();
        r.w = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);

        #if defined(RANDOM_SHUFFLE_STATE)
        if ((rng_state_shift % RngState::GetShuffleDistance()) == 0)
            state->Shuffle();
        #endif

        UnloadRngState(state, rng_state, rng_id);

        const auto block_x = 2 * blockDim.x;
        const auto block_y = blockDim.y;

        auto* sub_lattice = reinterpret_cast<Spin*>(shared_memory);
        auto* sub_cluster = reinterpret_cast<GpuLabelType*>(shared_memory + Ceil<64>(block_x * block_y * sizeof(Spin)));

        const auto tile_x = ((blockIdx.x + 1) == gridDim.x ? extent_0 - blockIdx.x * block_x : block_x);
        const auto tile_y = ((blockIdx.y + 1) == gridDim.y ? extent_1 - blockIdx.y * block_y : block_y);

        const auto tx = 2 * threadIdx.x;
        const auto ty = threadIdx.y;
        const auto x = blockIdx.x * block_x + tx;
        const auto y = blockIdx.y * block_y + ty;

        const auto active = (x < extent_0) && (y < extent_1);

        // Load the sub-lattice for the current block and initialize cluster.
        if (active)
        {
            sub_lattice[ty * block_x + tx] = lattice[y * extent_0 + x];
            sub_lattice[ty * block_x + tx + 1] = lattice[y * extent_0 + x + 1];

            sub_cluster[ty * block_x + tx] = ty * tile_x + tx;
            sub_cluster[ty * block_x + tx + 1] = ty * tile_x + tx + 1;
        }
        else
        {
            sub_lattice[ty * block_x + tx] = 0;
            sub_lattice[ty * block_x + tx + 1] = 0;

            sub_cluster[ty * block_x + tx] = 0;
            sub_cluster[ty * block_x + tx + 1] = 0;
        }

        // Connect 1-direction.
        if (active &&
            (sub_lattice[ty * block_x + tx] & 0x1) == (sub_lattice[ty * block_x + tx + 1] & 0x1) &&
            r.x < p_add)
        {
            sub_lattice[ty * block_x + tx] |= 0x2;
        }

        if (active && (tx + 1) < (tile_x - 1) &&
            (sub_lattice[ty * block_x + tx + 1] & 0x1) == (sub_lattice[ty * block_x + tx + 2] & 0x1) &&
            r.y < p_add)
        {
            sub_lattice[ty * block_x + tx + 1] |= 0x2;
        }

        // Connect 2-direction.
        if (ty < (tile_y - 1))
        {
            if (active &&
                (sub_lattice[ty * block_x + tx] & 0x1) == (sub_lattice[(ty + 1) * block_x + tx] & 0x1) &&
                r.z < p_add)
            {
                sub_lattice[ty * block_x + tx] |= 0x4;
            }

            if (active &&
                (sub_lattice[ty * block_x + tx + 1] & 0x1) == (sub_lattice[(ty + 1) * block_x + tx + 1] & 0x1) &&
                r.w < p_add)
            {
                sub_lattice[ty * block_x + tx + 1] |= 0x4;
            }
        }

        // Reduce labels: we can include all threads here as links have
        // been established for sites inside the sub-lattice only, and
        // hence data access remains inside the sub-lattice.
        const auto idx_even = ty * block_x + tx + (ty & 1);
        const auto idx_odd = ty * block_x + tx + ((ty + 1) & 1);
        bool labels_changed = true;
        while (__any(labels_changed))
        {
            labels_changed = false;

            // 1-direction reduction (even).
            if (sub_lattice[idx_even] & 0x2)
            {
                const auto a = sub_cluster[idx_even];
                const auto b = sub_cluster[idx_even + 1];
                if (a != b)
                {
                    const auto ab = min(a, b);
                    sub_cluster[idx_even] = ab;
                    sub_cluster[idx_even + 1] = ab;
                    labels_changed = true;
                }
            }

            // 2-drection reduction (even).
            if (sub_lattice[idx_even] & 0x4)
            {
                const auto a = sub_cluster[idx_even];
                const auto b = sub_cluster[idx_even + block_x];
                if (a != b)
                {
                    const auto ab = min(a, b);
                    sub_cluster[idx_even] = ab;
                    sub_cluster[idx_even + block_x] = ab;
                    labels_changed = true;
                }
            }

            // 1-direction reduction (odd).
            if (sub_lattice[idx_odd] & 0x2)
            {
                const auto a = sub_cluster[idx_odd];
                const auto b = sub_cluster[idx_odd + 1];
                if (a != b)
                {
                    const auto ab = min(a, b);
                    sub_cluster[idx_odd] = ab;
                    sub_cluster[idx_odd + 1] = ab;
                    labels_changed = true;
                }
            }

            // 2-drection reduction (odd).
            if (sub_lattice[idx_odd] & 0x4)
            {
                const auto a = sub_cluster[idx_odd];
                const auto b = sub_cluster[idx_odd + block_x];
                if (a != b)
                {
                    const auto ab = min(a, b);
                    sub_cluster[idx_odd] = ab;
                    sub_cluster[idx_odd + block_x] = ab;
                    labels_changed = true;
                }
            }
        }

        // Write back cluster.
        if (active)
        {
            const auto a = sub_cluster[ty * block_x + tx] % tile_x;
            const auto b = sub_cluster[ty * block_x + tx] / tile_x;
            cluster[y * extent_0 + x] = (blockIdx.y * block_y + b) * extent_0 + blockIdx.x * block_x + a;
        }

        if (active)
        {
            const auto a = sub_cluster[ty * block_x + tx + 1] % tile_x;
            const auto b = sub_cluster[ty * block_x + tx + 1] / tile_x;
            cluster[y * extent_0 + x + 1] = (blockIdx.y * block_y + b) * extent_0 + blockIdx.x * block_x + a;
        }
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::AssignLabels(Context& context, Lattice<2>& lattice, const float p_add)
    {
        const auto& extent = cluster.Extent();
        HipContext& gpu = static_cast<HipContext&>(context);

        // Configure kernel launch.
        const auto grid = dim3{(extent[0] + tile_size[0] - 1) / tile_size[0], (extent[1] + tile_size[1] - 1) / tile_size[1], 1};
        const auto block = dim3{tile_size[0] / 2, tile_size[1], 1};

        const auto shared_mem_rng_state_bytes = sizeof(RngState);
        const auto shared_mem_lattice_bytes = Ceil<64>(tile_size[0] * tile_size[1] * sizeof(Lattice<2>::Spin));
        const auto shared_mem_cluster_bytes = tile_size[0] * tile_size[1] * sizeof(GpuLabelType);
        const auto shared_mem_bytes = std::max(shared_mem_rng_state_bytes, shared_mem_lattice_bytes + shared_mem_cluster_bytes);

        SafeCall(hipSetDevice(gpu.Id()));

        AssignLabels_Kernel<<<grid, block, shared_mem_bytes>>>(lattice.RawGpuPointer(), extent[0], extent[1],
            gpu_cluster.get(), gpu_rng_state.get(), 100 * drand48(), p_add);

        gpu.Synchronize();
    }

    template <typename LabelType>
    __device__
    void Merge(LabelType* ptr, LabelType a, LabelType b);

    template <typename Spin, typename LabelType, typename RngState>
    __global__
    void MergeLabels_Kernel(const Spin* lattice, const std::uint32_t extent_0, const std::uint32_t extent_1,
        const std::uint32_t block_x, const std::uint32_t block_y,
        LabelType* cluster, RngState* rng_state, const std::uint32_t rng_state_shift, const float p_add)
    {
        const auto rng_id = (blockIdx.y * gridDim.x + blockIdx.x + rng_state_shift) % (gridDim.x * gridDim.y);
        auto* state = LoadRngState(rng_state, rng_id);

        auto r = float2{};
        state->Update();
        r.x = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);
        state->Update();
        r.y = 2.3283064370807974e-10f * state->Get(threadIdx.y * blockDim.x + threadIdx.x);

        UnloadRngState(state, rng_state, rng_id);

        const auto tile_x = ((blockIdx.x + 1) == gridDim.x ? extent_0 - blockIdx.x * block_x : block_x);
        const auto tile_y = ((blockIdx.y + 1) == gridDim.y ? extent_1 - blockIdx.y * block_y : block_y);

        // 1-direction.
        if (threadIdx.x < tile_y)
        {
            const auto index = (blockIdx.y * block_y + threadIdx.x) * extent_0 + blockIdx.x * block_x + tile_x - 1;
            const auto index_p1 = index + 1 - ((blockIdx.x + 1) == gridDim.x ? extent_0 : 0);

            if (lattice[index] == lattice[index_p1] && r.x < p_add)
            {
                LabelType a = cluster[index];
                LabelType b = cluster[index_p1];
                if (a != b)
                    Merge(cluster, a, b);
            }
        }

        // 2-direction.
        if (threadIdx.x < tile_x)
        {
            const auto index = (blockIdx.y * block_y + (tile_y - 1)) * extent_0 + blockIdx.x * block_x + threadIdx.x;
            const auto index_p1 = index + extent_0 - ((blockIdx.y + 1) == gridDim.y ? extent_0 * extent_1 : 0);

            if (lattice[index] == lattice[index_p1] && r.y < p_add)
            {
                LabelType a = cluster[index];
                LabelType b = cluster[index_p1];
                if (a != b)
                    Merge(cluster, a, b);
            }
        }
    }

    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::MergeLabels(Context& context, Lattice<2>& lattice, const float p_add)
    {
        const auto& extent = cluster.Extent();
        HipContext& gpu = static_cast<HipContext&>(context);

        // Configure kernel launch.
        const auto grid = dim3{(extent[0] + tile_size[0] - 1) / tile_size[0], (extent[1] + tile_size[1] - 1) / tile_size[1], 1};
        const auto block = dim3{std::min(WavefrontSize, std::max(tile_size[0], tile_size[1])), 1, 1};
        const auto shared_mem_bytes = sizeof(RngState);

        SafeCall(hipSetDevice(gpu.Id()));

        MergeLabels_Kernel<<<grid, block, shared_mem_bytes>>>(lattice.RawGpuPointer(), extent[0], extent[1],
            tile_size[0], tile_size[1],
            gpu_cluster.get(), gpu_rng_state.get(), 100 * drand48(), p_add);

        gpu.Synchronize();
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
    void ResolveLabels_Kernel(LabelType* cluster, const std::uint32_t extent_0, const std::uint32_t extent_1)
    {
        const auto block_x = blockDim.x;
        const auto block_y = blockDim.y;

        const auto tx = threadIdx.x;
        const auto ty = threadIdx.y;
        const auto x = blockIdx.x * block_x + tx;
        const auto y = blockIdx.y * block_y + ty;

        const auto active = (x < extent_0) && (y < extent_1);

        if (active)
        {
            const auto index = y * extent_0 + x;

            auto c = cluster[index];
            while (c != cluster[c])
                c = cluster[c];

            atomicExch(cluster + index, c);
        }
    }

    // Resolve all label equivalences
    template <template <DeviceName> typename RNG, DeviceName Target>
    void SwendsenWang_2D<RNG, Target>::ResolveLabels(Context& context)
    {
        const auto& extent = cluster.Extent();
        HipContext& gpu = static_cast<HipContext&>(context);

        // Configure kernel launch: this kernel will be mostly latency and memory bandwidth bound
        // -> we need many threads per block.
        const auto& tile_size = std::array<std::uint32_t, 2>{64, 8};
        const auto grid = dim3{(extent[0] + tile_size[0] - 1) / tile_size[0], (extent[1] + tile_size[1] - 1) / tile_size[1], 1};
        const auto block = dim3{tile_size[0], tile_size[1], 1};

        SafeCall(hipSetDevice(gpu.Id()));

        ResolveLabels_Kernel<<<grid, block>>>(gpu_cluster.get(), extent[0], extent[1]);

        gpu.Synchronize();
    }

    template <typename Spin, typename LabelType>
    __global__
    void FlipClusters_Kernel(Spin* lattice, const std::uint32_t extent_0, const std::uint32_t extent_1,
        const LabelType* cluster)
    {
        const auto block_x = blockDim.x;
        const auto block_y = blockDim.y;

        const auto tx = threadIdx.x;
        const auto ty = threadIdx.y;
        const auto x = blockIdx.x * block_x + tx;
        const auto y = blockIdx.y * block_y + ty;

        const auto active = (x < extent_0) && (y < extent_1);
        if (active)
        {
            lattice[y * extent_0 + x] ^= (cluster[y * extent_0 + x] & 0x1);
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
        const auto& extent = cluster.Extent();
        HipContext& gpu = static_cast<HipContext&>(context);

        // Configure kernel launch: this kernel will be mostly latency and memory bandwidth bound
        // -> we need many threads per block.
        const auto& tile_size = std::array<std::uint32_t, 2>{64, 8};
        const auto grid = dim3{(extent[0] + tile_size[0] - 1) / tile_size[0], (extent[1] + tile_size[1] - 1) / tile_size[1], 1};
        const auto block = dim3{tile_size[0], tile_size[1], 1};

        SafeCall(hipSetDevice(gpu.Id()));

        FlipClusters_Kernel<<<grid, block>>>(lattice.RawGpuPointer(), extent[0], extent[1],
            gpu_cluster.get());

        gpu.Synchronize();
    }

    // Explicit template instantiation.
    template void SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>::InitializeGpuRngState();
    template void SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>::InitializeGpuCluster(const LabelType);
    template void SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>::AssignLabels(Context&, Lattice<2>&, const float);
    template void SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>::MergeLabels(Context&, Lattice<2>&, const float);
    template void SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>::ResolveLabels(Context&);
    template void SwendsenWang_2D<LCG32, DeviceName::AMD_GPU>::FlipClusters(Context&, Lattice<2>&);
}

#endif