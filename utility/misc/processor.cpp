#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX   /* Do not define min and max in windows.h */
#include <windows.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

#include "processor.hpp"

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    std::set<CpuCore> CpuCoreTopology()
    {
        auto mapping = std::set<CpuCore>{};

#if defined(_WIN32)
        // First call to get the required buffer size.
        auto length = DWORD{};
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);

        // Now actually get the processor info.
        std::vector<std::byte> buffer(length);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length))
        {
            auto ptr = buffer.data();
            const auto end = ptr + length;
            auto physical_cpu_core = std::uint32_t{};

            while (ptr < end)
            {
                const auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
                if (info->Relationship == RelationProcessorCore)
                {
                    ++physical_cpu_core;

                    // Each GroupMask entry corresponds to one processor group.
                    const auto& groupMask = info->Processor.GroupMask[0];
                    const auto mask = static_cast<std::uint64_t>(groupMask.Mask);

                    for (std::uint32_t i = 0; i < 64; ++i)
                        if (mask & (std::uint64_t{ 1 } << i))
                            mapping.emplace(physical_cpu_core, i);
                }
                ptr += info->Size;
            }
        }
        else
        {
            std::cerr << "GetLogicalProcessorInformationEx failed. Error: " << GetLastError() << std::endl;
            std::cerr << "Fall back to trivial CPU topology." << std::endl;

            const auto num_cpus = std::thread::hardware_concurrency();
            for (std::uint32_t i = 0; i < num_cpus; ++i)
                mapping.emplace(i, i);
        }
#else
        auto cpuinfo = std::ifstream{"/proc/cpuinfo"};
        auto physical = std::int32_t{-1};
        auto logical = std::int32_t{-1};
        auto line = std::string{};

        while (std::getline(cpuinfo, line))
        {
            if (line.rfind("core id", 0) == 0)
                physical = std::stoi(line.substr(line.find(':') + 1));
            else if (line.rfind("processor", 0) == 0)
                logical = std::stoi(line.substr(line.find(':') + 1));

            if (physical >= 0 && logical >= 0)
            {
                mapping.emplace(physical, logical);
                physical = -1;
                logical = -1;
            }
        }
#endif

        return mapping;
    }

    void PinToCpuCore(const std::uint32_t cpu_core)
    {
#if defined(_WIN32)
        const auto my_set = static_cast<DWORD_PTR>(1ULL << cpu_core);
        SetThreadAffinityMask(GetCurrentThread(), my_set);
#else
        auto my_set = cpu_set_t{};
        CPU_ZERO(&my_set);
        CPU_SET(cpu_core, &my_set);
        sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#endif
    }
}

#undef XXX_NAMESPACE
