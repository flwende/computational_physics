#pragma once

#include <memory>
#include <optional>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class Awaitable
    {
        public:
            virtual ~Awaitable() = default;

            virtual void Wait() = 0;
    };

    template <typename T>
    class Future final
    {
        private:
            std::shared_ptr<T> value {};
            std::optional<Awaitable*> handle {};

        public:
            Future() = default;

            Future(std::shared_ptr<T>& value, Awaitable* handle) noexcept
                :
                value(std::move(value)), handle(handle)
            {}

            auto Valid() const noexcept { return static_cast<bool>(handle) && static_cast<bool>(value); }

            std::optional<T> Get() const
            {
                if (Valid())
                {
                    handle.value()->Wait();
                    return *value;
                }
                
                return {};
            }
    };
}

#undef XXX_NAMESPACE
