#pragma once

#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    // Replace the data pointed to by ptr by desired if and only if desired is smaller.
    template <typename T>
    T AtomicMin(volatile T* const ptr, const T desired)
    {
        static_assert(std::is_arithmetic_v<T> && sizeof(T) <= 8, "Error: unsupported type.");

        while (true)
        {
            // Store the value pointed to by ptr.
            T old = *ptr;

            // If this value is already lower than the assumed (desired) one, return it...
            if (old <= desired)
                return *ptr;

            // ...if not, try to replace it by the desired one.
            // The replacement is successful only if in the meantime the value pointed to by ptr
            // did not change. If so (__sync_bool...() returns true), return the old value pointed to by ptr.
            // Otherwise, try again: it might be possible that the new value pointed to by ptr is still
            // larger than the desired one.
            if (__sync_bool_compare_and_swap(ptr, old, desired))
                return old;
        }
    }
}
