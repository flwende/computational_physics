#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <numeric>

#include "accessor.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <typename T, std::uint32_t Dimension>
    class NonOwningMultiDimensionalArray
    {
        static_assert(Dimension > 0, "Error: Dimension must be larger than 0.");

        template <typename, std::uint32_t>
        friend class MultiDimensionalArray;

        protected:
            Accessor<T, Dimension> accessor;

        public:
            explicit NonOwningMultiDimensionalArray(T* external_ptr, const std::array<std::int32_t, Dimension>& extent) noexcept
                :
                accessor(external_ptr, extent)
            {}

            auto Initialized() const noexcept { return accessor.RawPointer() != nullptr; }
            const auto& Extent() const noexcept { return accessor.Extent(); }
            auto TotalElements() const noexcept { return accessor.TotalElements(); }

            T* RawPointer() noexcept { return accessor.RawPointer(); }
            const T* RawPointer() const noexcept { return accessor.RawPointer(); }

            auto operator[](const std::int32_t index) -> decltype(accessor[0]) { return accessor[index]; }
            auto operator[](const std::int32_t index) const -> decltype(accessor[0]) { return accessor[index]; }

        protected:
            NonOwningMultiDimensionalArray() noexcept = default;
    };

    template <typename T, std::uint32_t Dimension>
    class MultiDimensionalArray final
    {
        using PointerType = std::unique_ptr<T[]>;

        private:
            PointerType data {};
            NonOwningMultiDimensionalArray<T, Dimension> span {};

        public:
            MultiDimensionalArray() noexcept = default;

            explicit MultiDimensionalArray(const std::array<std::int32_t, Dimension>& extent)
                :
                data(std::make_unique<T[]>(std::accumulate(std::begin(extent), std::end(extent), 1, std::multiplies<std::size_t>()))),
                span(data.get(), extent)
            {}
            
            MultiDimensionalArray(const MultiDimensionalArray&) = delete;
            MultiDimensionalArray& operator=(const MultiDimensionalArray& other) = delete;

            MultiDimensionalArray(MultiDimensionalArray&&) noexcept = default;
            MultiDimensionalArray& operator=(MultiDimensionalArray&&) noexcept = default;

            auto Initialized() const { return span.Initialized(); }
            const auto& Extent() const noexcept { return span.Extent(); }
            auto TotalElements() const noexcept { return span.TotalElements(); }

            T* RawPointer() noexcept { return span.RawPointer(); }
            const T* RawPointer() const noexcept { return span.RawPointer(); }

            auto operator[](const std::int32_t index) -> decltype(span[0]) { return span[index]; }
            auto operator[](const std::int32_t index) const -> decltype(span[0]) { return span[index]; }

            void Resize(const std::array<std::int32_t, Dimension>& extent)
            {
                MultiDimensionalArray tmp(extent);
                std::swap(*this, tmp);
            }

            auto DeepCopy() const
            {
                MultiDimensionalArray tmp(span.Extent());
                std::copy_n(RawPointer(), TotalElements(), tmp.RawPointer());
                return tmp;
            }
    };
}
