#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sched.h>

#include "thread_context.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class ThreadGroup
    {
        static constexpr std::uint32_t MaxManagedStackMemorySize = 65536; // This is 64kB, which should be good in most cases.

        protected:
            std::atomic<bool> active {false};
            std::atomic<std::uint32_t> managed_stack_memory_bytes {};
            std::uint32_t num_threads {};
            Barrier new_task {};
            Barrier task_done {};
            Barrier barrier;
            std::vector<std::thread> threads;
            std::function<void(ThreadContext&)> kernel;

        public:
            explicit ThreadGroup(const std::uint32_t num_threads)
                :
                num_threads(num_threads),
                new_task(num_threads),
                task_done(num_threads),
                barrier(num_threads)
            {
                PinThread(0); // The master thread always exists: pin it to core 0.

                active = true;
                threads.reserve(num_threads - 1);
                for (std::int32_t id = 1; id < num_threads; ++id)
                    threads.emplace_back([this] (const std::uint32_t thread_id) { Run(thread_id); }, id);

                // Give threads some time to spin up.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            ThreadGroup(const ThreadGroup&) = delete;
            ThreadGroup& operator=(const ThreadGroup&) = delete;

            ThreadGroup(ThreadGroup&& other) = delete;
            ThreadGroup& operator=(ThreadGroup&& other) = delete;

            virtual ~ThreadGroup()
            {
                active.store(false, std::memory_order_release);
                new_task.Signal();

                for (auto& thread : threads)
                    thread.join();
            }

            auto Size() const { return num_threads; }

            void SetManagedStackMemorySize(const std::uint32_t bytes)
            {
                if (bytes > MaxManagedStackMemorySize)
                    throw ManagedMemoryError("Managed stack memory size exceeded.");

                managed_stack_memory_bytes = bytes;
            }
            
            template <typename Func, typename ...Args>
            void Execute(Func&& func, Args&&... args)
            {
                ThreadContext context(num_threads, 0, barrier);

                // Bind arguments to 'func'. The 1st parameter will be the thread context.
                kernel = std::bind(std::forward<Func>(func),
                    std::placeholders::_1,
                    std::forward<Args>(args)...);

                new_task.Signal();

                if (managed_stack_memory_bytes > 0)
                    ExecuteInNewStackFrame(kernel, context);
                else
                    kernel(context);

                task_done.Wait();
            }

            void Synchronize() { task_done.Wait(); }

        protected:
            template <typename Func>
            void ExecuteInNewStackFrame(Func&& func, ThreadContext& context)
            {
                std::byte* ptr = reinterpret_cast<std::byte*>(alloca(managed_stack_memory_bytes));
                context.StackMemory().Register(ptr, managed_stack_memory_bytes);

                func(context);

                context.StackMemory().Reset();
            }

            void Run(const std::uint32_t thread_id)
            {
                PinThread(thread_id % std::thread::hardware_concurrency());

                ThreadContext context(num_threads, thread_id, barrier);

                while (active.load(std::memory_order_acquire))
                {
                    new_task.Wait();
                    if (!active.load(std::memory_order_acquire))
                        break;

                    if (managed_stack_memory_bytes > 0)
                        ExecuteInNewStackFrame(kernel, context);
                    else
                        kernel(context);

                    // Mark the task as completed.
                    task_done.Signal();
                }
            }

            void PinThread(const std::uint32_t thread_id)
            {
                cpu_set_t my_set;
                CPU_ZERO(&my_set);
                CPU_SET(thread_id, &my_set);
                sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
            }
    };
}

#undef XXX_NAMESPACE
