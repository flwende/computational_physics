#if defined __HIPCC__

__device__
std::uint32_t Get(const std::int32_t index) const 
{ 
    return state[index];
}

__device__
void Load(const LCG32_State<WaveFrontSize>& other)
{
    state[threadIdx.x] = other.state[threadIdx.x];
    a[threadIdx.x] = other.a[threadIdx.x];
    c[threadIdx.x] = other.c[threadIdx.x];

    if (threadIdx.x == 0)
        iteration = other.iteration;
}

__device__
void Unload(LCG32_State<WaveFrontSize>& other) const
{
    other.state[threadIdx.x] = state[threadIdx.x];
    other.a[threadIdx.x] = a[threadIdx.x];
    other.c[threadIdx.x] = c[threadIdx.x];

    if (threadIdx.x == 0)
        other.iteration = iteration;
}

__device__
void Shuffle()
{
    if (threadIdx.x == 0)
        iteration += shuffle_distance;

    constexpr std::int32_t m = WaveFrontSize - 1;
    const std::int32_t shuffle_val = state[iteration & m] + (iteration & 1 ? 0 : 1);

    a[threadIdx.x] = a[(threadIdx.x + shuffle_val) & m];
    c[threadIdx.x] = c[(threadIdx.x + shuffle_val) & m];
}

__device__
void Update()
{
    state[threadIdx.x] = a[threadIdx.x] * state[threadIdx.x] + c[threadIdx.x];
}

__device__
static constexpr std::uint32_t ShuffleDistance()
{ 
    return shuffle_distance;
}

#endif