#include "thread_group.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    SingleTaskThreadGroup::SingleTaskThreadGroup(const std::uint32_t num_threads)
        :
        num_threads(num_threads), new_task(num_threads), task_done(num_threads), barrier(num_threads)
    {
        // The master thread always exists: pin it to core 0.
        PinToCpuCore(0);

        threads.reserve(num_threads);
        for (std::uint32_t i = 0; i < num_threads; ++i)
            threads.emplace_back([this, thread_id = i] ()
                {
                    PinThread(thread_id);

                    all_threads_up.Signal();

                    Run(thread_id);
                });

        all_threads_up.Wait();

        active.store(true, std::memory_order_relaxed);
    }

    SingleTaskThreadGroup::~SingleTaskThreadGroup()
    {
        // Signal to threads that the threads group is going down.
        active.store(false, std::memory_order_release);
        new_task.Signal();

        // Wake up the async (master) thread.
        async_execution.store(true, std::memory_order_release);
        async_cv.notify_one();
    }

    void SingleTaskThreadGroup::Wait()
    {
        if (async_execution)
        {
            std::unique_lock lock(async_lock);
            async_cv.wait(lock, [this] () { return async_complete.load(std::memory_order_acquire); });

            async_execution.store(false, std::memory_order_relaxed);
            async_complete.store(false, std::memory_order_relaxed);
        }
    }

    void SingleTaskThreadGroup::SetManagedStackMemorySize(const std::uint32_t bytes)
    {
        if (bytes > MaxManagedStackMemorySize)
            throw ManagedMemoryError("Maxium managed stack memory size exceeded.");

        managed_stack_memory_bytes = bytes;
    }

    void SingleTaskThreadGroup::Run(const std::uint32_t thread_id)
    {
        auto context = ThreadContext{num_threads, thread_id, barrier};

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
            }
            // Regular worker threads.
            else
            {
                new_task.Wait();
            }

            // Is the thread group still active?
            if (!active.load(std::memory_order_acquire))
                break;

            // Do the work.
            if (managed_stack_memory_bytes > 0)
                ExecuteInNewStackFrame(kernel, context);
            else
                kernel(context);

            // Mark the task as completed.
            task_done.Signal();

            // Async (master) thread.
            if (thread_id == 0)
            {
                // At this point all other worker threads are done too, so we can signal back to the
                // actual master thread we are done.
                async_complete.store(true, std::memory_order_release);
                async_cv.notify_one();
            }
        }
    }

    void SingleTaskThreadGroup::PinThread(const std::uint32_t thread_id) const
    {
        auto cpu_core = [this, thread_id] ()
            {
                auto cpu_cores = CpuCoreTopology();
                const auto physical_cpu_cores = cpu_cores
                    | std::ranges::views::transform([] (const auto& core) { return core.physical_id; });
                const auto physical_cpu_core_ids = std::set<std::uint32_t>{physical_cpu_cores.begin(), physical_cpu_cores.end()};
                const auto num_physical_cpu_cores = physical_cpu_core_ids.size();
                const auto num_logical_cpu_cores = cpu_cores.size();

                if (num_threads < num_physical_cpu_cores)
                {
                    // Assign one phyical core per thread, excluding core 0.
                    return cpu_cores.lower_bound({GetElement(physical_cpu_core_ids, thread_id + 1), 0})->logical_id;
                }
                else if (num_threads < (num_logical_cpu_cores - 1))
                {
                    // Remove physical core 0 from the list.
                    std::erase_if(cpu_cores, [] (const auto& core) { return core.physical_id == 0; });
                }

                // Round robin mapping.
                return GetElement(cpu_cores, thread_id % num_logical_cpu_cores).logical_id;
            };

        PinToCpuCore(cpu_core());
    }
}

#undef XXX_NAMESPACE
