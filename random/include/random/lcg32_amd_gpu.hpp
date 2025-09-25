#if defined __HIPCC__

__device__
auto Get(const std::int32_t index) const
{ 
    return state[index];
}

__device__
void Load_1d(const LCG32_State<WaveFrontSize>& other)
{
    const auto index = threadIdx.x;

    state[index] = other.state[index];
    a[index] = other.a[index];
    c[index] = other.c[index];

    if (index == 0)
        iteration = other.iteration;
}

__device__
void Load(const LCG32_State<WaveFrontSize>& other)
{
    const auto index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    state[index] = other.state[index];
    a[index] = other.a[index];
    c[index] = other.c[index];

    if (index == 0)
        iteration = other.iteration;
}

__device__
void Unload_1d(LCG32_State<WaveFrontSize>& other) const
{
    const auto index = threadIdx.x;

    other.state[index] = state[index];
    other.a[index] = a[index];
    other.c[index] = c[index];

    if (index == 0)
        other.iteration = iteration;
}

__device__
void Unload(LCG32_State<WaveFrontSize>& other) const
{
    const auto index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    other.state[index] = state[index];
    other.a[index] = a[index];
    other.c[index] = c[index];

    if (index == 0)
        other.iteration = iteration;
}

__device__
void Shuffle_1d()
{
    const auto index = threadIdx.x;

    if (index == 0)
        iteration += ShuffleDistance;

    constexpr auto m = WaveFrontSize - 1;
    const auto shuffle_val = state[iteration & m] + (iteration & 1 ? 0 : 1);

    a[index] = a[(index + shuffle_val) & m];
    c[index] = c[(index + shuffle_val) & m];
}

__device__
void Shuffle()
{
    const std::int32_t index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    if (index == 0)
        iteration += ShuffleDistance;

    constexpr auto m = WaveFrontSize - 1;
    const auto shuffle_val = state[iteration & m] + (iteration & 1 ? 0 : 1);

    a[index] = a[(index + shuffle_val) & m];
    c[index] = c[(index + shuffle_val) & m];
}

__device__
void Update_1d()
{
    const auto index = threadIdx.x;

    state[index] = a[index] * state[index] + c[index];
}

__device__
void Update()
{
    const auto index = threadIdx.z * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    state[index] = a[index] * state[index] + c[index];
}

#endif