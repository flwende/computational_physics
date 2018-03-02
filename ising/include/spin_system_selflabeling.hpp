// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SPIN_SYSTEM_SELFLABELING_HPP)
#define SPIN_SYSTEM_SELFLABELING_HPP

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
	//! \brief Connected component labeling (ccl) based on an idea of Coddington and Baillie within tiles
	//!
	//! References:
	//! \n
	//! * C. F. Baillie and P. D. Coddington,
	//!            "Cluster identification algorithms for spin models - sequential and parallel",
	//!            1991.
	//! \n
	//! * F. Wende and Th. Steinke,
	//!            "Swendsen-Wang Multi-Cluster Algorithm for the 2D/3D Ising Model on Xeon Phi and GPU",
	//!            SC'13 Proceedings, Article No. 83 ACM New York, NY, USA, 2013
	//!
	//! \n
	//! General idea:
	//! \n
	//! 1. load tile into local memory: bit 0 is either set (spin up) or not (spin down).
	//! \n
	//! 2. for each site in the inner of the tile use bits 1 and 2 to encode whether its
	//! neighboring site in 0- and 1-direction is aligned (has the same orientation) and whether
	//! both of the two should be connected (depending on the value of p_add).
	//! \n
	//! 3. initialize the cluster associated with this tile so that all labels are unique (e.g. use the 1-D index).
	//! \n
	//! 4. go through all sites within the tile and for each one assign to it and its connected neighbor(s)
	//! (see step 2) the minimum label. Iterate this step as long as labels change.
	//! \n
	//! 5. translate local labels to global labels.
	//! Label L is mapped to L' = ((n_offset[1] + b) * n[0] + n_offset[0] + a, where
	//! a = (L % n_0) and b = (L / n_0) and n_0 is either N_0 or n_sub[0].
	//!
	//! \tparam N_0 tile size in 0-direction
	//! \param p_add probability for adding aligned nearest neighbor sites to the cluster
	//! \param n_offset tile offset w.r.t. to (0, 0)
	//! \param n_sub extent of the tile
	template <std::size_t N_0>
	void spin_system<2>::ccl_selflabeling(const float p_add, const std::size_t (&n_offset) [2], const std::size_t (&n_sub) [2])
	{
		const std::uint32_t thread_id = omp_get_thread_num();
		// possible compiler optimization: N_0 has default value 0.
		// if the extent of the tile in 0-direction equals chunk[0] (= multiple of the SIMD width),
		// the compiler can maybe apply some SIMD related optimizations
		const std::size_t ii_max = (N_0 == 0 ? n_sub[0] : N_0);
		const std::size_t jj_max = n_sub[1];
		// local copy of the tile
		std::uint32_t l[jj_max][ii_max];
		// local cluster
		std::uint32_t c[jj_max][ii_max];
		// random numbers
		float buffer[chunk[0]];
		// temporaries
		std::uint32_t l_1[ii_max];

		// step 1
		for (std::size_t jj = 0; jj < jj_max; ++jj)
		{
			#pragma omp simd
			for (std::size_t ii = 0; ii < ii_max; ++ii)
			{
				l[jj][ii] = lattice[(n_offset[1] + jj) * n[0] + n_offset[0] + ii];
			}
		}

		// step 2: 0-direction -> set bit 1 if connected
		for (std::size_t jj = 0; jj < jj_max; ++jj)
		{
			rng[thread_id]->next_float(buffer, chunk[0]);

			for (std::size_t ii = 0; ii < (ii_max - 1); ++ii)
			{
				l_1[ii] = l[jj][ii + 1];
			}
			l_1[ii_max - 1] = 0x2;

			#pragma omp simd
			for (std::size_t ii = 0; ii < ii_max; ++ii)
			{
				std::uint32_t l_0 = l[jj][ii];
				if (l_0 == l_1[ii] && buffer[ii] < p_add)
				{
					l_0 |= 0x2;
				}
				l[jj][ii] = l_0;
			}
		}

		// step 2: 1-direction -> set bit 2 if connected
		for (std::size_t jj = 0; jj < (jj_max - 1); ++jj)
		{
			rng[thread_id]->next_float(buffer, chunk[0]);
			#pragma omp simd
			for (std::size_t ii = 0; ii < ii_max; ++ii)
			{
				std::uint32_t l_0 = l[jj][ii];
				if ((l_0 & 0x1) == (l[jj + 1][ii] & 0x1) && buffer[ii] < p_add)
				{
					l_0 |= 0x4;
				}
				l[jj][ii] = l_0;
			}
		}

		// step 3: use 1-D index for the initial labeling (unique)
		for (std::size_t jj = 0; jj < jj_max; ++jj)
		{
			#pragma omp simd
			for (std::size_t ii = 0; ii < ii_max; ++ii)
			{
				c[jj][ii] = jj * ii_max + ii;
			}
		}

		// step 4
		bool break_loop = false;
		while (!break_loop)
		{
			break_loop = true;
			for (std::size_t jj = 0; jj < jj_max; ++jj)
			{
				bool label_changes = true;
				while (label_changes)
				{
					#pragma omp simd
					for (std::size_t ii = 0; ii < ii_max; ++ii)
					{
						l_1[ii] = (l[jj][ii] & 0x2);
					}

					label_changes = false;
					for (std::size_t ii = 0; ii < (ii_max - 1); ++ii)
					{
						if (l_1[ii])
						{
							const std::uint32_t a = c[jj][ii];
							const std::uint32_t b = c[jj][ii + 1];
							if (a != b)
							{
								// replace both labels by their minimum
								const std::uint32_t ab = std::min(a, b);
								c[jj][ii] = ab;
								c[jj][ii + 1] = ab;
								label_changes = true;
							}
						}
					}

					if (label_changes)
					{
						break_loop = false;
					}
				}

				if (jj == (jj_max - 1))
				{
					// no next row in 1-direction
					continue;
				}

				#pragma omp simd
				for (std::size_t ii = 0; ii < ii_max; ++ii)
				{
					l_1[ii] = (l[jj][ii] & 0x4);
				}

				std::uint32_t counter = 0;
				#pragma omp simd reduction(+ : counter)
				for (std::size_t ii = 0; ii < ii_max; ++ii)
				{
					if (l_1[ii])
					{
						const std::uint32_t a = c[jj][ii];
						const std::uint32_t b = c[jj + 1][ii];
						if (a != b)
						{
							// replace both labels by their minimum
							const std::uint32_t ab = std::min(a, b);
							c[jj][ii] = ab;
							c[jj + 1][ii] = ab;
							counter++;
						}
					}
				}

				if (counter)
				{
					break_loop = false;
				}
			}
		}

		// step 5: translate local to global labels
		for (std::size_t jj = 0; jj < jj_max; ++jj)
		{
			for (std::size_t ii = 0; ii < ii_max; ++ii)
			{
				const std::uint32_t a = c[jj][ii] % ii_max;
				const std::uint32_t b = c[jj][ii] / ii_max;
				cluster[(n_offset[1] + jj) * n[0] + (n_offset[0] + ii)] = (n_offset[1] + b) * n[0] + (n_offset[0] + a);
			}
		}
	}
}

#undef XXX_NAMESPACE

#endif
