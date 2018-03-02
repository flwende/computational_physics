// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(LCG32_HPP)
#define LCG32_HPP

#include <cstdint>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Implementation of class random using the 32-bit linear congruential generator (lcg)
	//! a la NUMERICAL RECIPES
	//!
	//! Reference:
	//! \n
	//! * Saul Teukolsky, William H. Press and William T. Vetterling,
	//!            "Numerical Recipes in C: The Art of Scientific Computing, 3rd Edition"
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class alignas(simd::alignment) lcg32 : public random
	{
		//! SIMD width for data type std::uint32_t on the selected platform
		static constexpr std::size_t simd_width = simd::type<std::uint32_t>::width;
		//! Internal state of simd_width many concurrent lcgs
		std::uint32_t state[simd_width];
		//! Lcg parameters a (see NUMERICAL RECIPES)
		std::uint32_t a[simd_width];
		//! Lcg parameters c (see NUMERICAL RECIPES)
		std::uint32_t c[simd_width];

		//! Internal buffer size: we generate chunks of random numbers (multiple of simd_width)
		static constexpr std::size_t buffer_size = 4 * simd_width;
		//! Internal buffer
		std::uint32_t buffer[buffer_size];
		//! Current element in the buffer to be accessed next
		std::uint32_t current_element;
		//! Number of updates of the internal state already performed
		std::uint32_t num_updates;

		//! Number of updates of the internal state after which the concurrent lcgs exchange their parameters
		static constexpr std::size_t shuffle_distance = 15;

	public:

		//! \brief Constructor
		//!
		//! \param seed
		lcg32(const std::uint32_t seed = 1)
				:
				current_element(buffer_size - 1),
				num_updates(0)
		{
			// parameters are taken from NUMERICAL RECIPES
			static constexpr std::uint32_t num_parameters = 5;
			static constexpr std::uint32_t parameters[num_parameters][2] = {
					{1372383749U, 1289706101U},
					{2891336453U, 1640531513U},
					{2024337845U, 797082193U},
					{32310901U, 626627237U},
					{29943829U, 1013904223U}};

			std::uint32_t seed_init;
			#pragma omp critical (RANDOM_LOCK)
			{
				// random assignment of parameters to concurrent lcgs
				srand48(seed);
				for (std::size_t i = 0; i < simd_width; ++i)
				{
					const std::size_t idx = static_cast<std::size_t>(1000.0 * drand48()) % num_parameters;
					a[i] = parameters[idx][0];
					c[i] = parameters[idx][1];
				}

				// generate the seed value for calling init()
				seed_init = static_cast<std::uint32_t>(0xEFFFFFFFU * drand48()) + 1;
			}

			init(seed_init);
		}

		//! \brief Destructor
		~lcg32()
		{
			;
		}

		//! \brief Set up the internal state of the lcg
		//!
		//! \param seed
		void init(const std::uint32_t seed)
		{
			#pragma omp critical (RANDOM_LOCK)
			{
				// random initialization of the first lcg
				srand48(seed);
				state[0] = a[0] * (static_cast<std::uint32_t>(0xEFFFFFFFU * drand48()) + 1) + c[0];
			}

			// the n-th lcg is initialized using the state of the (n - 1)-th lcg
			for (std::size_t i = 1; i < simd_width; ++i)
			{
				state[i] = a[i] * state[i - 1] + c[i];
			}
		}

		//! \brief Get the next random unsigned 32-bit integer
		//!
		//! \return random integer over [0, 0xFFFFFFFF]
		std::uint32_t next_uint32()
		{
			if ((++current_element) == buffer_size)
			{
				// buffer is empty: refill it
				update();
				current_element = 0;
			}

			return buffer[current_element];
		}

		//! \brief Get the next random float
		//!
		//! \return random float over [0.0, 1.0]
		float next_float()
		{
			return (next_uint32() * 2.3283064370807974E-10F);
		}

		//! \brief Get the next n random unsigned 32-bit integers
		//!
		//! \param ptr output buffer
		//! \param n number of random integers
		void next_uint32(std::uint32_t* ptr, const std::size_t n)
		{
			// write as many numbers as possible to ptr directly
			const std::size_t i_max = (n / buffer_size) * buffer_size;
			for (std::size_t i = 0; i < i_max; i += buffer_size)
			{
				update(&ptr[i]);
			}

			// take the rest from the internal buffer
			for (std::size_t i = i_max; i < n; ++i)
			{
				ptr[i] = next_uint32();
			}
		}

		//! \brief Get the next n random floats
		//!
		//! \param ptr output buffer
		//! \param n number of random floats
		void next_float(float* ptr, const std::size_t n)
		{
			// reinterpret the output buffer type for calling the update method
			std::uint32_t* i_ptr = reinterpret_cast<std::uint32_t*>(ptr);
			next_uint32(i_ptr, n);

			// convert to float over [0.0, 1.0]
			for (std::size_t i = 0; i < n; ++i)
			{
				ptr[i] = i_ptr[i] * 2.3283064370807974E-10F;
			}
		}

	private:

		//! \brief Update the internal state
		//!
		//! In case of RANDOM_SHUFFLE_STATE is defined, the lcg states are exchanged every shuffle_distance-th update.
		//!
		//! \param ptr (optional) output buffer, to intercept writing to the internal buffer
		void update(std::uint32_t* ptr = nullptr)
		{
			// if no output buffer is specified, use the internal buffer
			if (ptr == nullptr)
			{
				ptr = buffer;
			}

			#if defined(RANDOM_SHUFFLE_STATE)
			// exchange lcg states at random
			if (((++num_updates) % shuffle_distance) == 0)
			{
				constexpr std::size_t m = simd_width - 1;
				const std::size_t shuffle_val = state[num_updates & m] + (num_updates & 1 ? 0 : 1);

				#pragma omp simd
				for (std::size_t ii = 0; ii < simd_width; ++ii)
				{
					ptr[ii] = a[(ii + shuffle_val) & m];
				}

				#pragma omp simd
				for (std::size_t ii = 0; ii < simd_width; ++ii)
				{
					a[ii] = ptr[ii];
				}

				#pragma omp simd
				for (std::size_t ii = 0; ii < simd_width; ++ii)
				{
					ptr[ii] = c[(ii + shuffle_val) & m];
				}

				#pragma omp simd
				for (std::size_t ii = 0; ii < simd_width; ++ii)
				{
					c[ii] = ptr[ii];
				}
			}
			#endif

			// generate buffer_size many random integers
			for (std::size_t i = 0; i < buffer_size; i += simd_width)
			{
				#pragma omp simd
				for (std::size_t ii = 0; ii < simd_width; ++ii)
				{
					// update internal state
					state[ii] = a[ii] * state[ii] + c[ii];
					// push to output buffer
					ptr[i + ii] = state[ii];
				}
			}
		}
	};
}

#undef XXX_NAMESPACE

#endif
