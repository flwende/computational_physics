#pragma once

#include <memory>
#include <optional>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    class Awaitable
    {
        public:
            virtual ~Awaitable() noexcept = default;

            virtual void Wait() = 0;
    };

    template <typename T>
    class Future final
    {
        private:
            std::shared_ptr<T> value {};
            Awaitable* handle {};

        public:
            Future() noexcept = default;

            Future(std::shared_ptr<T>&& value, Awaitable* handle) noexcept
                :
                value(std::move(value)), handle(handle)
            {}

            std::optional<T> Get() const
            {
                if (handle)
                    handle->Wait();

                if (value)
                    return *value;
                
                return {};
            }
    };
}

#undef XXX_NAMESPACE
