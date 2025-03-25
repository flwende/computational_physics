#pragma once

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <thread>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class Barrier final
    {
        public:
            Barrier(const std::int32_t expected = 1)
            {
                Reset(expected);
            }

            void Reset(const std::int32_t new_expected)
            {   
                std::unique_lock<std::mutex> lock(cv_mutex);

                // Reset the state.
                value = 0;
                expected = new_expected;

                // Move forward to the next epoch and unblock all waiting threads.
                ++epoch;
                cv.notify_all();            
            }
            
            void Wait()
            {
                std::unique_lock<std::mutex> lock(cv_mutex);
                
                if (++value == expected)
                {
                    value = 0;
                    // All waiting threads will use a value of 'epoch' that is 1
                    // this new value. This guarantees that this thread will wait at
                    // the barrier in the next round if the barrier is reached again
                    // immediately after notify_all.
                    ++epoch;
                    cv.notify_all();
                }
                else
                {
                    // Make a copy of 'epoch' and wait for it to change.
                    const std::uint32_t current_epoch = epoch;
                    cv.wait(lock, [this, current_epoch] () { return epoch != current_epoch; });
                }
            }

            // Special semantics, e.g., for master-worker pattern.
            void Signal() { Wait(); }
            
        private:
            std::uint32_t expected{1};
            std::atomic<std::uint32_t> value{0}, epoch{0};
            std::condition_variable cv;
            std::mutex cv_mutex;
    };

    class LockFreeBarrier final
    {
        public:
            LockFreeBarrier(const std::int32_t expected = 1)
            {
                Reset(expected);
            }

            void Reset(const std::int32_t new_expected)
            {   
                // This does not conflict with the implementation of 'Wait()'.
                expected = new_expected;
                value = 0;
                ++epoch;
            }
            
            void Wait()
            {                
                // Make a copy of 'epoch' and wait for it to change.
                const std::uint32_t current_epoch = epoch;

                if (++value == expected)
                {
                    value = 0;
                    ++epoch;
                }
                else
                {
                    while (epoch == current_epoch)
                        std::this_thread::yield();
                }
            }

            // Special semantics, e.g., for master-worker pattern.
            void Signal() { Wait(); }
            
        private:
            std::atomic<std::uint32_t> expected{1}, value{0}, epoch{0};
    };
}