#pragma once

#include <cassert>
#include <cstdint>

#include "context.hpp"
#include "memory/managed_memory.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class ThreadContext final : public Context
    {
        private:
            ManagedMemory stack_memory {};
            Barrier& barrier;

        public:
            ThreadContext(const std::uint32_t group_size, const std::uint32_t id, Barrier& barrier)
                :
                Context(group_size, id), barrier(barrier)
            {}

            auto NumThreads() const { return GroupSize(); }
            auto ThreadId() const { return Id(); }

            ManagedMemory& StackMemory() { return stack_memory; }

            void Synchronize() override { barrier.Wait(); }
    };
}

#undef XXX_NAMESPACE
