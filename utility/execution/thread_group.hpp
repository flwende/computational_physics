#pragma once

#include <iostream>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <ranges>
#include <thread>
#include <vector>

#include "thread_context.hpp"
#include "future/future.hpp"
#include "misc/processor.hpp"
#include "misc/set_operations.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class SingleTaskThreadGroup final : public Awaitable
    {
        // Execute a single task (kernel) across multiple CPU threads in a cooperative manner.
        //
        // General: Tasks (kernels) are expected to take a 'context' reference as a first argument.
        //
        //      The context allows threads executing the kernel to ask for their logical thread ID and
        //      thread group size, synchronize with all other threads in the group, and allocate
        //      dynamic stack memory.
        //
        // Master thread: Always exists and executes on CPU core 0 in any case.
        //      Updates the kernel object that should be executed, and signals to the worker threads
        //      to start executing the kernel. The master thread participates the execution unless
        //      the execution is asynchronous.
        //
        // Worker thread(s): 'num_threads - 1' many threads are spawned in the constructor plus one
        //      master sibling thread for asynchronous kernel execution. All worker threads execute
        //      on CPU cores != 0 (if possible) to not interfere with the master thread.
        //
        //      For asynchronous kernel execution it is important to spawn less worker threads than
        //      there are CPU cores so that none of the worker threads interfers with the master thread.
        //      Otherwise, the master thread would slow down the worker thread(s) and vice versa.
        //
        // Synchronous kernel execution: the master thread synchronizes with all worker threads after
        //      kernel execution and before returning control back to the callee.
        //      The master sibling thread is inactive.
        //
        // Asynchronous kernel execution: the master thread hands over its role to its sibling thread
        //      and returns control back to the callee. The master thread calls Wait() to synchronize
        //      with its sibling thread, which in turn synchronizes with all other worker threads after
        //      kernel execution.
        //
        // Both synchronous and asynchronous kernel calls return a pointer to a ThreadPool object
        // that can be used to set up a Future for synchronization in the asynchronous case.
        // In the synchronous case, the returned pointer is a NULL pointer.

        static constexpr auto MaxManagedStackMemorySize = std::uint32_t{65536}; // This is 64kB, which should be good in most cases.

        private:
            std::atomic<bool> active {false};
            std::atomic<bool> async_execution {false};
            std::atomic<bool> async_complete {false};
            std::atomic<std::uint32_t> managed_stack_memory_bytes {};
            std::condition_variable async_cv {};
            std::mutex async_lock {};
            std::uint32_t num_threads {};
            Barrier all_threads_up {};
            Barrier new_task {};
            Barrier task_done {};
            Barrier barrier;
            std::vector<std::jthread> threads;
            std::function<void(ThreadContext&)> kernel;

        public:
            explicit SingleTaskThreadGroup(const std::uint32_t num_threads);

            SingleTaskThreadGroup(const SingleTaskThreadGroup&) = delete;
            SingleTaskThreadGroup& operator=(const SingleTaskThreadGroup&) = delete;

            SingleTaskThreadGroup(SingleTaskThreadGroup&& other) = delete;
            SingleTaskThreadGroup& operator=(SingleTaskThreadGroup&& other) = delete;

            ~SingleTaskThreadGroup();

            void Wait() override;

            auto Size() const noexcept { return num_threads; }

            void SetManagedStackMemorySize(const std::uint32_t bytes);
            
            template <typename Func, typename ...Args>
            auto Execute(Func&& func, Args&&... args) -> SingleTaskThreadGroup*
            {
                // Bind arguments to 'func'. The 1st parameter will be the thread context.
                kernel = std::bind(std::forward<Func>(func),
                    std::placeholders::_1,
                    std::forward<Args>(args)...);

                // Let all worker threads know there is new work.
                new_task.Signal();

                // Participate the kernel execution.
                auto context = ThreadContext{num_threads, 0, barrier};
                if (managed_stack_memory_bytes > 0)
                    ExecuteInNewStackFrame(kernel, context);
                else
                    kernel(context);

                // Synchronize with worker threads: blocking!
                task_done.Wait();

                return {};
            }

            template <typename Func, typename ...Args>
            auto AsyncExecute(Func&& func, Args&&... args) -> SingleTaskThreadGroup*
            {
                // Bind arguments to 'func'. The 1st parameter will be the thread context.
                kernel = std::bind(std::forward<Func>(func),
                    std::placeholders::_1,
                    std::forward<Args>(args)...);

                // Let the async (master) thread know there is new work.
                async_execution.store(true, std::memory_order_release);
                async_cv.notify_one();

                // Let all worker threads know there is new work.
                new_task.Signal();

                return this;
            }

        private:
            template <typename Func>
            void ExecuteInNewStackFrame(Func&& func, ThreadContext& context)
            {
                auto ptr = reinterpret_cast<std::byte*>(alloca(managed_stack_memory_bytes));
                context.StackMemory().Register(ptr, managed_stack_memory_bytes);

                func(context);

                context.StackMemory().Reset();
            }

            void Run(const std::uint32_t thread_id);

            void PinThread(const std::uint32_t thread_id) const;
    };
}

#undef XXX_NAMESPACE
