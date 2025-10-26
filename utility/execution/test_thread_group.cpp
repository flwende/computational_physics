#include <iostream>
#include <cstdint>

#include "thread_group.hpp"

using namespace cp;

void foo(ThreadContext& context, int a)
{
    const auto num_threads = context.NumThreads();
    const auto thread_id = context.ThreadId();

    for (std::int32_t i = 0; i < num_threads; ++i)
    {
        context.Synchronize();
        if (i == thread_id)
            std::cout << context.ThreadId() << ", " << a << std::endl;
    }
}

void bar(ThreadContext& context, int a, float b)
{
    const auto num_threads = context.NumThreads();
    const auto thread_id = context.ThreadId();

    for (std::int32_t i = 0; i < num_threads; ++i)
    {
        context.Synchronize();
        if (i == thread_id)
            std::cout << thread_id << ", " << a << ", " << b << std::endl;
    }

    context.Synchronize();

    for (std::int32_t i = 0; i < num_threads; ++i)
    {
        context.Synchronize();
        if (i == thread_id)
            std::cout << "After barrier 1" << std::endl;
    }

    context.Synchronize();

    for (std::int32_t i = 0; i < num_threads; ++i)
    {
        context.Synchronize();
        if (i == thread_id)
            std::cout << "After barrier 2" << std::endl;
    }
}

int main()
{
    auto tg = ThreadGroup{2};

    tg.Execute(foo, 42);
    tg.Execute(bar, 23, 4.3f);

    return 0;
}