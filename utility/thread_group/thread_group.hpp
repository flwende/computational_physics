#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>
#include <sched.h>

#include "context.hpp"
#include "synchronization/barrier.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class ThreadContext final : public Context
    {
        public:
            ThreadContext(const std::int32_t group_size, const std::int32_t id, LockFreeBarrier& barrier)
                :
                Context(group_size, id),
                barrier(barrier)
            {}

            const std::int32_t NumThreads() const { return GroupSize(); }
            const std::int32_t ThreadId() const { return Id(); }

            void Synchronize() override { barrier.Wait(); }
        
        private:
            LockFreeBarrier& barrier;
    };

    class ThreadGroup
    {
        public:
            ThreadGroup(const std::int32_t num_threads)
                :
                num_threads(num_threads),
                new_task(num_threads),
                task_done(num_threads),
                barrier(num_threads)
            {
                PinThread(0);

                active = true;
                threads.reserve(num_threads - 1);
                for (std::int32_t i = 1; i < num_threads; ++i)
                    threads.emplace_back([this] (const std::int32_t thread_id) { Run(thread_id); }, i);
            }

            ThreadGroup(const ThreadGroup&) = delete;
            ThreadGroup& operator=(const ThreadGroup&) = delete;

            ThreadGroup(ThreadGroup&& other) = delete;
            ThreadGroup& operator=(ThreadGroup&& other) = delete;

            virtual ~ThreadGroup()
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

                active = false;
                new_task.Signal();

                for (auto& thread : threads)
                    thread.join();
            }

            auto Size() const { return num_threads; }
            
            template <typename Func, typename ...Args>
            void Execute(Func&& func, Args&&... args)
            {
                ThreadContext context{num_threads, 0, barrier};

                kernel = std::bind(std::forward<Func>(func),
                    std::placeholders::_1, // thread context
                    std::forward<Args>(args)...);

                new_task.Signal();

                kernel(context);

                task_done.Wait();
            }

            void Synchronize() { task_done.Wait(); }

        protected:
            void Run(const std::int32_t thread_id)
            {
                PinThread(thread_id);

                ThreadContext context{num_threads, thread_id, barrier};

                while (active)
                {
                    new_task.Wait();
                    if (!active)
                        break;

                    // Execute the kernel.
                    kernel(context);

                    // Mark the task as completed.
                    task_done.Signal();
                }
            }

            void PinThread(const std::int32_t thread_id)
            {
                cpu_set_t my_set;
                CPU_ZERO(&my_set);
                CPU_SET(thread_id, &my_set);
                sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
            }

            std::atomic<bool> active{false};
            std::int32_t num_threads;
            LockFreeBarrier new_task;
            LockFreeBarrier task_done;
            LockFreeBarrier barrier;
            std::vector<std::thread> threads;
            std::function<void(ThreadContext&)> kernel;
    };
}

#undef XXX_NAMESPACE
