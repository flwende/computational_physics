#pragma once

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class Context
    {
        public:
            Context(const std::int32_t group_size, const std::int32_t id)
                :
                group_size(group_size),
                id(id)
            {}

            const std::int32_t GroupSize() const { return group_size; }
            const std::int32_t Id() const { return id; }

            virtual void Synchronize() = 0;
        
        protected:
            const std::int32_t group_size;
            const std::int32_t id;
    };
}

#undef XXX_NAMESPACE
