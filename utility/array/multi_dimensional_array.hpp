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
    class MultiDimensionalArray
    {
        static_assert(Dimension > 0, "Error: Dimension must be larger than 0.");

        public:
            MultiDimensionalArray() = default;
            
            MultiDimensionalArray(const std::array<std::int32_t, Dimension>& extent)
                :
                extent(extent),
                data(new T[std::accumulate(std::begin(extent), std::end(extent), 1, std::multiplies<std::size_t>())]),
                accessor(data.get(), extent)
            {}
            
            void Resize(const std::array<std::int32_t, Dimension>& extent)
            {
                MultiDimensionalArray tmp{extent};
                std::swap(*this, tmp);
            }

            bool Initialized() const { return data.get() != nullptr; }
            const auto& Extent() const { return extent; }

            T* RawPointer() { return data.get(); }
            const T* RawPointer() const { return data.get(); }

            auto operator[](const std::int32_t index) { return accessor[index]; }
            const auto operator[](const std::int32_t index) const { return accessor[index]; }

        protected:
            std::array<std::int32_t, Dimension> extent;
            std::unique_ptr<T[]> data;
            Accessor<T, Dimension> accessor;
    };
}
