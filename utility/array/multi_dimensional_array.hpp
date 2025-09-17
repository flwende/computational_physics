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
    template <typename T, std::int32_t Dimension>
    class NonOwningMultiDimensionalArray
    {
        static_assert(Dimension > 0, "Error: Dimension must be larger than 0.");

        protected:
            Accessor<T, Dimension> accessor;

        public:
            explicit NonOwningMultiDimensionalArray(T* external_ptr, const std::array<std::int32_t, Dimension>& extent)
                :
                accessor(external_ptr, extent)
            {}

            bool Initialized() const { return accessor.RawPointer() != nullptr; }
            const auto& Extent() const { return accessor.Extent(); }
            auto TotalElements() const { return accessor.TotalElements(); }

            T* RawPointer() { return accessor.RawPointer(); }
            const T* RawPointer() const { return accessor.RawPointer(); }

            inline auto operator[](const std::int32_t index) -> decltype(accessor[0]) { return accessor[index]; }
            inline auto operator[](const std::int32_t index) const -> decltype(accessor[0]) { return accessor[index]; }

        protected:
            NonOwningMultiDimensionalArray() = default;
    };

    template <typename T, std::int32_t Dimension>
    class MultiDimensionalArray : public NonOwningMultiDimensionalArray<T, Dimension>
    {
        using Base = NonOwningMultiDimensionalArray<T, Dimension>;
        using PointerType = std::unique_ptr<T[]>;

        protected:
            PointerType data;

        public:
            MultiDimensionalArray() = default;

            explicit MultiDimensionalArray(const std::array<std::int32_t, Dimension>& extent)
                :
                MultiDimensionalArray(std::make_unique<T[]>(std::accumulate(std::begin(extent), std::end(extent), 1, std::multiplies<std::size_t>())), extent)
            {};
            
            MultiDimensionalArray(const MultiDimensionalArray&) = delete;
            MultiDimensionalArray& operator=(const MultiDimensionalArray& other) = delete;

            MultiDimensionalArray(MultiDimensionalArray&&) noexcept = default;
            MultiDimensionalArray& operator=(MultiDimensionalArray&&) noexcept = default;

            void Resize(const std::array<std::int32_t, Dimension>& extent)
            {
                MultiDimensionalArray tmp(extent);
                std::swap(*this, tmp);
            }

            MultiDimensionalArray DeepCopy() const
            {
                using Base::Extent;
                using Base::TotalElements;
                using Base::RawPointer;

                MultiDimensionalArray tmp(Extent());
                std::copy_n(std::begin(RawPointer()), TotalElements(), tmp.RawPointer());
                return tmp;
            }

        protected:
            MultiDimensionalArray(PointerType&& data, const std::array<std::int32_t, Dimension>& extent)
                :
                Base(data.get(), extent),
                data(std::move(data))
            {};
    };
}
