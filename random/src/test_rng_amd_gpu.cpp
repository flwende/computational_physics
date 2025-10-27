#if defined(__HIPCC__)

#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "device/device.hpp"
#include "random/random.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

// This value should be equal to the wavefront size of the GPU.
constexpr auto WavefrontSize = AMD_GPU::WavefrontSize();
constexpr auto Buffersize = WavefrontSize;

// Dynamic shared memory for GPU kernels.
extern __shared__ std::byte shared_memory[];

template <typename RngState>
__device__
auto* LoadRngState(const RngState* rng_state)
{
    const auto rng_id = std::uint32_t{blockIdx.x * blockDim.y + threadIdx.y};
    auto* shared_rng_state = reinterpret_cast<RngState*>(shared_memory + rng_id * sizeof(RngState));
    shared_rng_state->Load_1d(rng_state[rng_id]);
    return shared_rng_state;
}

template <typename RngState>
__device__
void UnloadRngState(const RngState* shared_rng_state, RngState* rng_state)
{
    const auto rng_id = std::uint32_t{blockIdx.x * blockDim.y + threadIdx.y};
    shared_rng_state->Unload_1d(rng_state[rng_id]);
}

template <typename RngState>
__global__
void Kernel(RngState* rng_state, float* output, const std::uint32_t iterations)
{
    const auto rng_id = std::uint32_t{blockIdx.x * blockDim.y + threadIdx.y};
    auto* state = LoadRngState(rng_state);
    auto* random_numbers = output + rng_id * Buffersize;

    for (std::uint32_t i = 0; i < iterations; i += RngState::GetShuffleDistance())
    {
        for (std::uint32_t ii = 0; ii < RngState::GetShuffleDistance(); ++ii)
        {
            state->Update_1d();
            random_numbers[threadIdx.x] = 2.3283064370807974E-10F * state->Get(threadIdx.x);
        }
        #if defined(RANDOM_SHUFFLE_STATE)
        state->Shuffle_1d();
        #endif
    }

    UnloadRngState(state, rng_state);
}

template <template <DeviceName> typename RNG, DeviceName Target>
std::pair<double, std::vector<float>> Benchmark(const std::uint32_t reporting_id, const std::pair<std::size_t, std::size_t> iterations)
{
    static_assert(Target == DeviceName::AMD_GPU, "Target must be AMD_GPU.");

    AMD_GPU gpu;
    const auto num_thread_blocks = gpu.Concurrency();
    const auto rngs_per_thread_block = 256 / WavefrontSize;
    const auto num_rngs = num_thread_blocks * rngs_per_thread_block;

    std::cout << "rngs_per_thread_block: " << rngs_per_thread_block << std::endl;
    std::cout << "num_rngs: " << num_rngs << std::endl;

    SafeCall(hipSetDevice(gpu.DeviceId()));

    // Allocate RNG state on the host.
    using RngState = typename RNG<DeviceName::AMD_GPU>::State;
    std::vector<RngState> host_rng_state(num_rngs);
    for (std::uint32_t i = 0; i < num_rngs; ++i)
        host_rng_state[i].Init(i + 1);

    GpuPointer<RngState> gpu_rng_state;
    {
        RngState* ptr {};
        SafeCall(hipMalloc(&ptr, num_rngs * sizeof(RngState)));
        SafeCall(hipMemcpy(ptr, host_rng_state.data(), num_rngs * sizeof(RngState), hipMemcpyHostToDevice));
        gpu_rng_state.reset(ptr);
    }

    GpuPointer<float> gpu_random_numbers;
    {
        float* ptr {};
        SafeCall(hipMalloc(&ptr, num_rngs * Buffersize * sizeof(float)));
        gpu_random_numbers.reset(ptr);
    }

    // Configure kernel launch.
    const auto grid = dim3{num_thread_blocks, 1, 1};
    const auto block = dim3{WavefrontSize, rngs_per_thread_block, 1};
    const auto shared_mem_bytes = rngs_per_thread_block * sizeof(RngState); // Per thread block.

    // Warmup.
    gpu.Execute([&] (auto& context)
        {
            const auto [warmup_iterations, _] = iterations;
            Kernel<RngState><<<grid, block, shared_mem_bytes>>>(gpu_rng_state.get(), gpu_random_numbers.get(),
                (warmup_iterations + (num_rngs * Buffersize - 1)) / (num_rngs * Buffersize));
            context.Synchronize();
        });

    // Benchmark.
    const auto [_, benchmark_iterations] = iterations;
    const auto starttime = std::chrono::high_resolution_clock::now();
    {
        gpu.Execute(Kernel<RngState>, grid, block, shared_mem_bytes,
            gpu_rng_state.get(), gpu_random_numbers.get(),
            (benchmark_iterations + (num_rngs * Buffersize - 1)) / (num_rngs * Buffersize));
        gpu.Synchronize();
    }
    const auto endtime = std::chrono::high_resolution_clock::now();
    const auto elapsed_time_s = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0E-6;

    // Extract random numbers from the GPU.
    std::vector<float> random_numbers(Buffersize);
    SafeCall(hipMemcpy(random_numbers.data(), gpu_random_numbers.get() + reporting_id * Buffersize, Buffersize * sizeof(float), hipMemcpyDeviceToHost));

    return {elapsed_time_s, random_numbers};
}

// Explicit instantiation.
template std::pair<double, std::vector<float>> Benchmark<LCG32, DeviceName::AMD_GPU>(const std::uint32_t, const std::pair<std::size_t, std::size_t>);

#endif
