#pragma once

#include <iostream>
#include <algorithm>
#include <array>
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
        std::array<T, M> Extract(const std::array<T, N>& a)
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

        public:
            Accessor() = default;

            Accessor(T* ptr, const std::array<std::int32_t, Dimension>& extent)
                :
                extent(extent),
                ptr(ptr)
            {}
            
            const auto& Extent() const { return extent; }

            T* RawPointer() { return ptr; }
            const T* RawPointer() const { return ptr; }

            ReturnType operator[](const std::int32_t index)
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

            ConstReturnType operator[](const std::int32_t index) const
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

        private:
            std::array<std::int32_t, Dimension> extent;
            T* ptr;
    };
}
