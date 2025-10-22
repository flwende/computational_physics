#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#if defined (_WIN32)
#define NOMINMAX   /* Do not define min and max in windows.h */
#include <windows.h>
#else
#include <sched.h>
#endif

#include "thread_context.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class ThreadGroup
    {
        static constexpr auto MaxManagedStackMemorySize = std::uint32_t{65536}; // This is 64kB, which should be good in most cases.

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
                num_threads(num_threads), new_task(num_threads), task_done(num_threads), barrier(num_threads)
            {
                PinThread(0); // The master thread always exists: pin it to core 0.

                threads.reserve(num_threads - 1);
                for (std::uint32_t i = 1; i < num_threads; ++i)
                    threads.emplace_back([this] (const std::uint32_t thread_id) { Run(thread_id); }, i);

                active.store(true, std::memory_order_relaxed);
            }

            ThreadGroup(const ThreadGroup&) = delete;
            ThreadGroup& operator=(const ThreadGroup&) = delete;

            ThreadGroup(ThreadGroup&& other) = delete;
            ThreadGroup& operator=(ThreadGroup&& other) = delete;

            virtual ~ThreadGroup()
            {
                // Signal to threads that the threads group is going down.
                active.store(false, std::memory_order_release);
                new_task.Signal();

                for (auto& thread : threads)
                    thread.join();
            }

            auto Size() const noexcept { return num_threads; }

            void SetManagedStackMemorySize(const std::uint32_t bytes)
            {
                if (bytes > MaxManagedStackMemorySize)
                    throw ManagedMemoryError("Maxium managed stack memory size exceeded.");

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
                auto ptr = reinterpret_cast<std::byte*>(alloca(managed_stack_memory_bytes));
                context.StackMemory().Register(ptr, managed_stack_memory_bytes);

                func(context);

                context.StackMemory().Reset();
            }

            // This method is being called by worker threads.
            void Run(const std::uint32_t thread_id)
            {
                PinThread(thread_id % std::thread::hardware_concurrency());

                ThreadContext context(num_threads, thread_id, barrier);

                while (true)
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
#if defined (_WIN32)
                const auto my_set = static_cast<DWORD_PTR>(1ULL << thread_id);
                SetThreadAffinityMask(GetCurrentThread(), my_set);
#else
                auto my_set = cpu_set_t{};
                CPU_ZERO(&my_set);
                CPU_SET(thread_id, &my_set);
                sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#endif
            }
    };
}

#undef XXX_NAMESPACE
