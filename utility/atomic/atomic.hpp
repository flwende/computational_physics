#pragma once

#include <atomic>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    // Replace the data pointed to by ptr by desired if and only if desired is smaller.
    // The function returns the 'old' value pointed to by ptr.
    template <typename T>
    T AtomicMin(T* const ptr, const T desired) noexcept
    {
        static_assert(std::is_arithmetic_v<T> && sizeof(T) <= 8, "Error: unsupported type.");

        auto ref = std::atomic_ref<T>(*ptr);

        while (true)
        {
            // Store the 'old' value pointed to by ptr.
            auto old = ref.load();

            // If this value is already lower than the assumed (desired) one, return it...
            if (old <= desired)
                return old;

            // ...if not, try to replace it by the desired one.
            // The replacement is successful only if in the meantime the value pointed to by ptr
            // did not change. If so (compare_exchange_weak() returns true), return the old value pointed to by ptr.
            // Otherwise, try again: it might be possible that the new value pointed to by ptr is still
            // larger than the desired one.
            if (ref.compare_exchange_strong(old, desired))
                return old;
        }
    }
}
