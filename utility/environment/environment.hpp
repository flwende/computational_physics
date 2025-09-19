#pragma once

#include <cstdlib>
#include <string>
#include <type_traits>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE cp
#endif

namespace XXX_NAMESPACE
{
    template <typename T>
    T GetEnv(const std::string& name, const T& default_value = {})
    {
        try
        {
            const char* value = std::getenv(name.c_str());
            if (!value)
                return default_value;
            
            if constexpr (std::is_same_v<T, std::string>) 
            {
                return std::string(value);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return static_cast<T>(std::stoll(value));
            }
            else if constexpr (std::is_floating_point_v<T>)
            {
                return static_cast<T>(std::stod(value));
            }
            else
            {
                static_assert(false, "Unsupported type for environment variable.");
            }
        }
        catch (const std::exception&)
        {
            return default_value;
        }
    }
}

#undef XXX_NAMESPACE