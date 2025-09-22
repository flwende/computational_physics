#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <condition_variable>
#include <thread>

#include "environment/environment.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class Barrier final
    {
        private:
            std::atomic<std::uint32_t> expected {1}, value {0}, epoch {0}, active_waiters {0};
            std::atomic<bool> allow_entry {true};
            void (Barrier::* volatile implementation)() {};
            const std::uint32_t hardware_threads {};
            std::condition_variable cv {};
            std::mutex cv_mutex {};

        public:
            Barrier(const std::uint32_t expected = 1)
                :
                hardware_threads{GetEnv("NUM_THREADS", std::thread::hardware_concurrency())}
            {
                Reset(expected);
            }

            void Reset(const std::uint32_t new_expected)
            {
                assert(new_expected > 0 && "Barrier::expected must be > 0.");

                // Gate: no new threads are allowed to enter the barrier.
                allow_entry.store(false, std::memory_order_release);
                {
                    // Wait for active waiters to finish.
                    while (active_waiters.load(std::memory_order_acquire))
                        std::this_thread::yield();

                    // No thread(s) waiting. Update 'expected', 'value' and 'epoch'.
                    expected.store(new_expected, std::memory_order_relaxed);
                    value.store(0, std::memory_order_relaxed);
                    epoch.fetch_add(1, std::memory_order_release);

                    // Update 'implementation'.
                    implementation = (expected > hardware_threads ? &Barrier::CvWait : &Barrier::BusyWait);
                }
                allow_entry.store(true, std::memory_order_release);
            }
            
            void Wait()
            {
                while (true)
                {
                    // Wait while Reset is happening.
                    while (!allow_entry.load(std::memory_order_acquire))
                        std::this_thread::yield();

                    // Indicate this thread attempts to enter the barrier.
                    active_waiters.fetch_add(1, std::memory_order_acq_rel);

                    // Check gate again: if Reset started between the load and the increment, undo and retry.
                    if (allow_entry.load(std::memory_order_acquire))
                        break;

                    active_waiters.fetch_sub(1, std::memory_order_acq_rel);
                }

                assert(implementation && "Barrier::implementation not set.");

                (this->*implementation)();

                active_waiters.fetch_sub(1, std::memory_order_acq_rel);
            }

            // Special semantics, e.g., for master-worker pattern, or to let other threads know this thread
            // reached the barrier.
            void Signal() { Wait(); }

        private:
            void CvWait()
            {
                auto lock = std::unique_lock<std::mutex>{cv_mutex};

                // The last thread reaching the barrier will update the state and 'epoch'.
                if (value.fetch_add(1, std::memory_order_acq_rel) == (expected - 1))
                {
                    value.store(0, std::memory_order_relaxed);
                    epoch.fetch_add(1, std::memory_order_release);
                    cv.notify_all();
                }
                else
                {
                    // Make a copy of 'epoch' and wait for it to change.
                    cv.wait(lock, [this, current_epoch = epoch.load(std::memory_order_acquire)] ()
                        {
                            return epoch.load(std::memory_order_acquire) != current_epoch;
                        });
                }
            }
            
            // Use this method only if the number of hardware threads is lower than or equal to 'expected'.
            // Otherwise, busy waiting threads will compete with running threads for CPU cycles.
            void BusyWait()
            {
                // Make a copy of 'epoch' and wait for it to change.
                const auto current_epoch = epoch.load(std::memory_order_acquire);

                // The last thread reaching the barrier will update the state and 'epoch'.
                if (value.fetch_add(1, std::memory_order_acq_rel) == (expected - 1))
                {
                    value.store(0, std::memory_order_relaxed);
                    epoch.fetch_add(1, std::memory_order_release);
                }
                else
                {
                    while (epoch.load(std::memory_order_acquire) == current_epoch)
                        std::this_thread::yield();
                }
            }
    };
}