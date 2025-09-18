#pragma once

#include <array>
#include <cassert>
#include <cstdint>

#include "misc/accumulate.hpp"

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
            std::copy_n(std::begin(a), M, std::begin(b));
            return b;
        }
    }

    template <typename T, std::uint32_t Dimension>
    class Accessor final
    {
        static_assert(Dimension > 0, "Error: Dimension must be larger than 0.");

        using ReturnType = std::conditional_t<Dimension == 1, T&, Accessor<T, Dimension - 1>>;
        using ConstReturnType = std::conditional_t<Dimension == 1, const T&, Accessor<const T, Dimension - 1>>;

        private:
            T* ptr {};
            std::array<std::uint32_t, Dimension> extent {};
            std::size_t hyper_plane_size {};

        public:
            Accessor() = default;

            explicit Accessor(T* ptr, const std::array<std::uint32_t, Dimension>& extent) noexcept
                :
                ptr(ptr),
                extent(extent),
                hyper_plane_size(Accumulate<std::multiplies<std::size_t>>(extent | std::ranges::views::take(Dimension - 1), 1UL))
            {
                assert(ptr != nullptr && "Accessor expects a non-null pointer.");
            }

            Accessor(const Accessor&) noexcept = default;
            Accessor& operator=(const Accessor&) noexcept = default;

            Accessor(Accessor&& other) noexcept = default;
            Accessor& operator=(Accessor&& other) noexcept = default;

            auto& Extent() const noexcept { return extent; }
            auto Elements() const noexcept { return hyper_plane_size * extent[Dimension - 1]; }

            auto RawPointer() noexcept { return ptr; }
            auto RawPointer() const noexcept { return ptr; }

            ReturnType operator[](const std::int32_t index)
            {
                assert(index >= 0 && index < extent[Dimension - 1] && "Out of bounds array access.");

                if constexpr (Dimension == 1)
                {
                    return ptr[index];
                }
                else
                {
                    return ReturnType(&ptr[index * hyper_plane_size], Extract<Dimension - 1>(extent));
                }
            }

            ConstReturnType operator[](const std::int32_t index) const
            {
                assert(index >= 0 && index < extent[Dimension - 1] && "Out of bounds array access.");

                if constexpr (Dimension == 1)
                {
                    return ptr[index];
                }
                else
                {
                    return ConstReturnType(&ptr[index * hyper_plane_size], Extract<Dimension - 1>(extent));
                }
            }
    };
}
