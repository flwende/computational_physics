#include <iostream>
#include <cstdint>
#include <functional>
#include <vector>

#include "thread_group.hpp"

using namespace cp;

void foo(ThreadContext& context, int a)
{
    std::cout << context.ThreadId() << ", " << a << std::endl;
}

void bar(ThreadContext& context, int a, float b)
{
    std::cout << context.ThreadId() << ", " << a << ", " << b << std::endl;

    context.Synchronize();

    std::cout << "After barrier 1" << std::endl;

    context.Synchronize();

    std::cout << "After barrier 2" << std::endl;
}

int main()
{
    ThreadGroup tg(2);

    tg.Execute(foo, 42);
    tg.Execute(bar, 23, 4.3f);

    return 0;
}