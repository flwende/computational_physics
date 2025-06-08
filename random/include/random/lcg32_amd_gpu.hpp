#if defined __HIPCC__

__device__
std::uint32_t Get(const std::int32_t index) const 
{ 
    return state[index];
}

__device__
void Load(const LCG32_State<WaveFrontSize>& other)
{
    const std::int32_t index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    state[index] = other.state[index];
    a[index] = other.a[index];
    c[index] = other.c[index];

    if (index == 0)
        iteration = other.iteration;
}

__device__
void Unload(LCG32_State<WaveFrontSize>& other) const
{
    const std::int32_t index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    other.state[index] = state[index];
    other.a[index] = a[index];
    other.c[index] = c[index];

    if (index == 0)
        other.iteration = iteration;
}

__device__
void Shuffle()
{
    const std::int32_t index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    if (index == 0)
        iteration += shuffle_distance;

    constexpr std::int32_t m = WaveFrontSize - 1;
    const std::int32_t shuffle_val = state[iteration & m] + (iteration & 1 ? 0 : 1);

    a[index] = a[(index + shuffle_val) & m];
    c[index] = c[(index + shuffle_val) & m];
}

__device__
void Update()
{
    const std::int32_t index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    state[index] = a[index] * state[index] + c[index];
}

__device__
static constexpr std::uint32_t ShuffleDistance()
{ 
    return shuffle_distance;
}

#endif