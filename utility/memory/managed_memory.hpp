#pragma once

#include <iostream>

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include "memory.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    // This data type is intended to manage memory that got allocated externally, e.g., via alloca().
    // Use case: avoid variable length arrays as they are not part of the C++ standard.
    //
    // Example:
    //
    //  void bar(ManagedMemory& stack_memory, A a) {
    //      auto data = stack_memory.Get<std::uint32_t>(a.Elements()); 
    //      ..
    //      for (std::int32_t i = 0; i < a.Elements(); ++i)
    //          data[i] = ..;
    //      ..
    //  }
    // 
    //  void foo(std::vector<A>& v) {
    //      std::byte* ptr = alloca(1024);
    //      ManagedMemory stack_memory(ptr, 1024);
    //      for (auto& item : v)
    //          bar(stack_mem, item);
    //      ..
    //  }
    //
    class ManagedMemory final
    {
        public:
            template <typename T>
            class Pointer final
            {
                private:
                    T* ptr {};
                    std::uint32_t count {};
                    ManagedMemory& memory;

                public:
                    explicit Pointer(T* ptr, const std::uint32_t count, ManagedMemory& memory)
                        :
                        ptr(ptr), count(count), memory(memory)
                    {}

                    ~Pointer()
                    {
                        memory.Release(count * sizeof(T));
                    }

                    Pointer(const Pointer&) = delete;
                    Pointer& operator=(const Pointer&) = delete;

                    Pointer(Pointer&& other) noexcept
                        :
                        ptr(other.ptr),
                        count(other.count),
                        memory(other.memory)
                    {
                        if (this != &other)
                        {
                            other.ptr = {};
                            other.count =  {};
                        }
                    }

                    Pointer& operator=(Pointer&&) noexcept = default;

                    auto Get() { return ptr; }
                    const auto Get() const { return ptr; }
            };

        protected:
            std::byte* ptr {};
            std::uint32_t managed_memory_bytes {};
            std::uint32_t memory_bytes_used {};

        public:
            ManagedMemory() = default;

            ManagedMemory(std::byte* external_ptr, const std::uint32_t bytes)
                :
                ptr(external_ptr), managed_memory_bytes(bytes)
            {}

            ManagedMemory(const ManagedMemory&) = delete;
            ManagedMemory& operator=(const ManagedMemory&) = delete;

            ManagedMemory(ManagedMemory&&) = delete;
            ManagedMemory& operator=(ManagedMemory&&) = delete;

            void Register(std::byte* external_ptr, const std::uint32_t bytes)
            {
                ptr = external_ptr;
                managed_memory_bytes = bytes;
                memory_bytes_used = {};
            }

            void Reset()
            {
                ptr = {};
                managed_memory_bytes = {};
                memory_bytes_used = {};
            }

            template <typename T>
            auto Allocate(const std::uint32_t count)
            {
                // TODO: add alignment.
                return Pointer<T>(reinterpret_cast<T*>(GetBytes(count * sizeof(T))), count, *this);
            }

        private:
            std::byte* GetBytes(const std::uint32_t bytes)
            {
                assert(ptr != nullptr && "ManagedMemory has no memory registered.");
                assert((memory_bytes_used + bytes) <= managed_memory_bytes && "ManagedMemory does not manage enough memory.");

                memory_bytes_used += bytes;
                return ptr + (memory_bytes_used - bytes);
            }

            void Release(const std::uint32_t bytes)
            {
                assert(bytes <= memory_bytes_used && "ManagedMemory corrupted: more bytes released than managed.");

                memory_bytes_used -= bytes;
            }
    };

    class ManagedMemoryError : public std::runtime_error
    {
        using Base = std::runtime_error;

        public:
            ManagedMemoryError(const std::string& error_message) : Base(error_message) {}
    };
}

#undef XXX_NAMESPACE
