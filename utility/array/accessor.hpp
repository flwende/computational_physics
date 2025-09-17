#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    namespace
    {
        template <std::size_t M, typename T, std::size_t N>
        std::array<T, M> Extract(const std::array<T, N>& a) noexcept
        {
            static_assert(M <= N, "Error: M must be smaller or equal to N.");

            if constexpr (M == N)
                return a;
            
            std::array<T, M> b;
            std::copy(std::begin(a), std::begin(a) + M, std::begin(b));
            return b;
        }
    }

    template <typename T, std::int32_t Dimension>
    class Accessor final
    {
        static_assert(Dimension > 0, "Error: Dimension must be larger than 0.");

        using ReturnType = std::conditional_t<Dimension == 1, T&, Accessor<T, Dimension - 1>>;
        using ConstReturnType = std::conditional_t<Dimension == 1, const T&, Accessor<const T, Dimension - 1>>;

        private:
            T* ptr {};
            std::array<std::int32_t, Dimension> extent;
            std::size_t total_elements;

        public:
            Accessor() = default;

            Accessor(T* ptr, const std::array<std::int32_t, Dimension>& extent)
                :
                ptr(ptr),
                extent(extent),
                total_elements(std::accumulate(std::begin(extent), std::end(extent), 1, std::multiplies<std::size_t>()))
            {
                assert(ptr != nullptr && "Accessor expects a non-null pointer.");
            }

            Accessor(const Accessor&) = default;
            Accessor& operator=(const Accessor&) = default;

            Accessor(Accessor&& other) noexcept : ptr(other.ptr), extent(other.extent) { other.ptr = {}; other.extent = {}; other.total_elements = {}; }
            Accessor& operator=(Accessor&& other) noexcept
            {
                if (this != &other)
                {
                    ptr = other.ptr;
                    extent = other.extent;
                    total_elements = other.total_elements;
                    other.ptr = {};
                    other.extent = {};
                    other.total_elements = {};
                }
                return *this;
            }

            const auto& Extent() const { return extent; }
            auto TotalElements() const { return total_elements; }

            T* RawPointer() { return ptr; }
            const T* RawPointer() const { return ptr; }

            inline ReturnType operator[](const std::int32_t index)
            {
                if constexpr (Dimension == 1)
                {
                    return ptr[index];
                }
                else
                {
                    const std::size_t n = std::accumulate(std::begin(extent), std::end(extent) - 1, 1, std::multiplies<std::size_t>());
                    return {&ptr[index * n], Extract<Dimension - 1>(extent)};
                }
            }

            inline ConstReturnType operator[](const std::int32_t index) const
            {
                if constexpr (Dimension == 1)
                {
                    return ptr[index];
                }
                else
                {
                    const std::size_t n = std::accumulate(std::begin(extent), std::end(extent) - 1, 1, std::multiplies<std::size_t>());
                    return {&ptr[index * n], Extract<Dimension - 1>(extent)};
                }
            }
    };
}
