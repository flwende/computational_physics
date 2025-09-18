#pragma once

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    enum class MemoryKind : std::int32_t
    {
        Heap = 1,
        Stack = 2
    };

    template <typename, std::uint32_t, MemoryKind>
    class NonOwningMultiDimensionalArray;

    template <typename T, std::uint32_t Dimension>
    using VariableLengthArray = NonOwningMultiDimensionalArray<T, Dimension, MemoryKind::Stack>;
}

#undef XXX_NAMESPACE
