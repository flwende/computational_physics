#pragma once

#include <iostream>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <latch>
#include <mutex>
#include <thread>
#include <vector>

#if defined (_WIN32)
#define NOMINMAX   /* Do not define min and max in windows.h */
#include <windows.h>
#else
#include <sched.h>
#endif

#include "thread_context.hpp"
#include "future/future.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class ThreadGroup : public Awaitable
    {
        static constexpr auto MaxManagedStackMemorySize = std::uint32_t{65536}; // This is 64kB, which should be good in most cases.

        protected:
            std::atomic<bool> active {false};
            std::atomic<bool> async_execution {false};
            std::atomic<bool> async_complete {false};
            std::atomic<std::uint32_t> managed_stack_memory_bytes {};
            std::condition_variable async_cv {};
            std::mutex async_lock {};
            std::uint32_t num_threads {};
            Barrier new_task {};
            Barrier task_done {};
            Barrier barrier;
            std::vector<std::jthread> threads;
            std::function<void(ThreadContext&)> kernel;

        public:
            explicit ThreadGroup(const std::uint32_t num_threads)
                :
                num_threads(num_threads), new_task(num_threads), task_done(num_threads), barrier(num_threads)
            {
                auto all_threads_up = Barrier{num_threads + 1};

                PinToCpuCore(0); // The master thread always exists: pin it to core 0.

                threads.reserve(num_threads);
                for (std::uint32_t i = 0; i < num_threads; ++i)
                    threads.emplace_back([this, &all_threads_up, thread_id = i] ()
                    {
                        all_threads_up.Signal();
                        Run(thread_id);
                    });

                all_threads_up.Wait();

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

                // Wake up the async (master) thread.
                async_execution.store(true, std::memory_order_release);
                async_cv.notify_one();
            }

            auto Size() const noexcept { return num_threads; }

            void SetManagedStackMemorySize(const std::uint32_t bytes)
            {
                if (bytes > MaxManagedStackMemorySize)
                    throw ManagedMemoryError("Maxium managed stack memory size exceeded.");

                managed_stack_memory_bytes = bytes;
            }
            
            template <typename Func, typename ...Args>
            auto Execute(Func&& func, Args&&... args)
            {
                // Bind arguments to 'func'. The 1st parameter will be the thread context.
                kernel = std::bind(std::forward<Func>(func),
                    std::placeholders::_1,
                    std::forward<Args>(args)...);

                new_task.Signal();

                auto context = ThreadContext{num_threads, 0, barrier};
                if (managed_stack_memory_bytes > 0)
                    ExecuteInNewStackFrame(kernel, context);
                else
                    kernel(context);

                task_done.Wait();

                return this;
            }

            template <typename Func, typename ...Args>
            auto AsyncExecute(Func&& func, Args&&... args)
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

            void Wait() override
            {
                if (async_execution)
                {
                    std::unique_lock lock(async_lock);
                    async_cv.wait(lock, [this] () { return async_complete.load(std::memory_order_acquire); });

                    async_execution.store(false, std::memory_order_relaxed);
                    async_complete.store(false, std::memory_order_relaxed);
                }
            }

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
                auto context = ThreadContext{num_threads, thread_id, barrier};

                PinThread(thread_id % std::thread::hardware_concurrency());

                while (true)
                {
                    // Async (master) thread.
                    if (thread_id == 0)
                    {
                        std::unique_lock lock(async_lock);
                        async_cv.wait(lock, [this] ()
                            {
                                return async_execution.load(std::memory_order_acquire) &&
                                    !async_complete.load(std::memory_order_acquire);
                            });

                        if (!active.load(std::memory_order_acquire))
                            break;

                        if (managed_stack_memory_bytes > 0)
                            ExecuteInNewStackFrame(kernel, context);
                        else
                            kernel(context);

                        // Mark the task as completed.
                        task_done.Signal();

                        // At this point all other worker threads are done too, so we can signal back to the
                        // actual master thread we are done.
                        async_complete.store(true, std::memory_order_release);
                        async_cv.notify_one();
                    }
                    // Worker threads.
                    else
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
            }

            void PinThread(const std::uint32_t thread_id)
            {
                auto cpu_core = [this, thread_id] ()
                    {
                        const auto num_cpu_cores = std::thread::hardware_concurrency();
                        if (num_threads < (num_cpu_cores / 2))
                        {
                            return 1 + thread_id;
                        }
                        else if (num_threads < num_cpu_cores && thread_id == 0)
                        {
                            return ((num_threads + 1) % num_cpu_cores) / 2 + (num_cpu_cores / 2) * (num_threads % 2);
                        }
                        else
                        {
                            return ((thread_id + 1) % num_cpu_cores) / 2 + (num_cpu_cores / 2) * (thread_id % 2);
                        }
                    };

                PinToCpuCore(cpu_core());
            }

            void PinToCpuCore(const std::uint32_t cpu_core)
            {
#if defined (_WIN32)
                const auto my_set = static_cast<DWORD_PTR>(1ULL << cpu_core);
                SetThreadAffinityMask(GetCurrentThread(), my_set);
#else
                auto my_set = cpu_set_t{};
                CPU_ZERO(&my_set);
                CPU_SET(cpu_core, &my_set);
                sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#endif
            }


    };
}

#undef XXX_NAMESPACE
