#pragma once

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class Context
    {
        protected:
            const std::uint32_t group_size;
            const std::uint32_t id;

        public:
            explicit Context(const std::uint32_t group_size, const std::uint32_t id)
                :
                group_size(group_size),
                id(id)
            {}

            virtual ~Context() noexcept = default;

            const auto GroupSize() const { return group_size; }
            const auto Id() const { return id; }

            virtual void Synchronize() = 0;
    };
}

#undef XXX_NAMESPACE
