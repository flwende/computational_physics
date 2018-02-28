// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(RANDOM_HPP)
#define RANDOM_HPP

#include <cstdint>
#include <simd/simd.hpp>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    //! \brief Abstract random number generator class
    //!
    //! Uniform random integers over [0, 0xFFFFFFFF] and floats over [0.0, 1.0]
	class alignas(simd::alignment) random
    {

	public:

		virtual ~random() { ; }
		
		virtual void init(const std::uint32_t seed) = 0;

		virtual std::uint32_t next_uint32() = 0;

		virtual float next_float() = 0;

        virtual void next_uint32(std::uint32_t* ptr, const std::size_t n) = 0;

        virtual void next_float(float* ptr, const std::size_t n) = 0;
	};	
}

#include "lcg32.hpp"

#undef XXX_NAMESPACE

#endif
