#if defined __HIPCC__

#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "device/device.hpp"
#include "random/random.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

using namespace XXX_NAMESPACE;

// We can use a LockFreeBarrier here: master + worker thread.
static LockFreeBarrier barrier(2);

// This value should be equal to the wavefront size of the GPU.
constexpr std::int32_t Buffersize{AMD_GPU::WavefrontSize<std::uint32_t>()};

// Dynamic shared memory for GPU kernels.
extern __shared__ std::byte shared_memory[];

template <std::int32_t N>
__host__
__device__
std::int32_t Ceil(const std::int32_t value)
{
    return ((value + N - 1) / N) * N;
}

template <typename RngState>
__device__
RngState* LoadRngState(const RngState* rng_state)
{
    RngState* shared_rng_state = reinterpret_cast<RngState*>(shared_memory + blockIdx.x * Ceil<64>(sizeof(RngState)));
    shared_rng_state->Load(rng_state[blockIdx.x]);
    return shared_rng_state;
}

template <typename RngState>
__device__
void UnloadRngState(const RngState* shared_rng_state, RngState* rng_state)
{
    shared_rng_state->Unload(rng_state[blockIdx.x]);
}

template <typename RngState>
__global__
void Kernel(RngState* rng_state, float* output, const std::int32_t iterations)
{
    RngState* state = LoadRngState(rng_state);
    float* random_numbers = output + blockIdx.x * Buffersize;

    for (std::int32_t i = 0; i < iterations; i += RngState::ShuffleDistance())
    {
        for (std::int32_t ii = 0; ii < RngState::ShuffleDistance(); ++ii)
        {
            state->Update();
            random_numbers[threadIdx.x] = 2.3283064370807974e-10f * state->Get(threadIdx.x);
        }
        #if defined RANDOM_SHUFFLE_STATE
        state->Shuffle();
        #endif
    }

    UnloadRngState(state, rng_state);
}

template <template <DeviceName> typename RNG, DeviceName Target>
void BenchmarkImpl(std::vector<float>& output, const std::pair<std::size_t, std::size_t>& iterations, const std::int32_t reporting_id)
{
    using RngState = typename RNG<Target>::State;
        
    const typename Device<Target>::Type target;
    const std::uint32_t num_thread_blocks = target.Concurrency();
    std::vector<RngState> host_rng_state(num_thread_blocks);
    for (std::int32_t i = 0; i < num_thread_blocks; ++i)
        host_rng_state[i].Init(i + 1);

    SafeCall(hipSetDevice(target.DeviceID()));

    // Allocate GPU resources.
    GpuPointer<RngState> gpu_rng_state;
    {
        RngState* ptr{};
        SafeCall(hipMalloc(&ptr, num_thread_blocks * sizeof(RngState)));
        SafeCall(hipMemcpy(ptr, host_rng_state.data(), num_thread_blocks * sizeof(RngState), hipMemcpyHostToDevice));
        gpu_rng_state.reset(ptr);
    }

    GpuPointer<float> gpu_random_numbers;
    {
        float* ptr{};
        SafeCall(hipMalloc(&ptr, num_thread_blocks * Buffersize * sizeof(float)));
        gpu_random_numbers.reset(ptr);
    }

    // Configure kernel launch.
    const dim3 grid{num_thread_blocks, 1, 1};
    const dim3 block{AMD_GPU::WavefrontSize<std::uint32_t>(), 1, 1};
    const std::size_t shared_mem_bytes = num_thread_blocks * Ceil<64>(sizeof(RngState));

    // Warmup run.
    const auto [warmup_iterations, benchmark_iterations] = iterations;
    Kernel<<<grid, block, shared_mem_bytes>>>(gpu_rng_state.get(), gpu_random_numbers.get(),
        (warmup_iterations + (num_thread_blocks * Buffersize - 1)) / (num_thread_blocks * Buffersize));

    SafeCall(hipDeviceSynchronize());

    // Synchronize with the master thread before starting the Kernel loop (benchmark run).
    barrier.Wait();

    Kernel<<<grid, block, shared_mem_bytes>>>(gpu_rng_state.get(), gpu_random_numbers.get(),
        (benchmark_iterations + (num_thread_blocks * Buffersize - 1)) / (num_thread_blocks * Buffersize));

    SafeCall(hipDeviceSynchronize());

    // Signal back to the master thread.
    barrier.Wait();

    // Extract random numbers from the GPU.
    SafeCall(hipMemcpy(output.data(), gpu_random_numbers.get() + reporting_id * Buffersize, Buffersize * sizeof(float), hipMemcpyDeviceToHost));
}

template <template <DeviceName> typename RNG, DeviceName Target>
std::pair<double, std::vector<float>> Benchmark(const std::int32_t reporting_id, const std::pair<std::size_t, std::size_t> iterations)
{
    std::vector<float> random_numbers(Buffersize);
    std::thread worker = std::thread(BenchmarkImpl<RNG, Target>, std::ref(random_numbers), std::ref(iterations), reporting_id);

    // Synchronize with threads in the Kernel kernel.
    barrier.Wait();
    const auto starttime = std::chrono::high_resolution_clock::now();
    barrier.Wait();
    const auto endtime = std::chrono::high_resolution_clock::now();
    const double elapsed_time_s = std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count() * 1.0e-6;

    worker.join();

    return {elapsed_time_s, random_numbers};
}

// Explicit instantiation.
template std::pair<double, std::vector<float>> Benchmark<LCG32, DeviceName::AMD_GPU>(const std::int32_t, const std::pair<std::size_t, std::size_t>);

#endif
