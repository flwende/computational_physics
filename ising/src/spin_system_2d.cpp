// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <spin_system.hpp>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
    //! \brief Constructor
    //!
    //! \param n lattice extent
	spin_system<2>::spin_system(const std::size_t (&n) [2])
			:
			n{n[0], n[1]},
			lattice(std::vector<std::uint8_t>(n[0] * n[1])),
			cluster(std::vector<std::uint32_t>(n[0] * n[1])),
			rng(std::vector<random*>(omp_get_max_threads(), nullptr)),
			call_measurement_routine(true)
	{
        // initialize random number generators (using the thread IDs) and
        // the lattice (at random)
		#pragma omp parallel
		{
			const std::uint32_t thread_id = omp_get_thread_num();
			rng[thread_id] = new lcg32(1 + thread_id);

            // arbitrarily chosen buffer size
			const std::size_t buffer_size = std::min(128UL, n[0]);
			float buffer[buffer_size];

			#pragma omp for
			for (std::size_t j = 0; j < n[1]; ++j)
			{
				for (std::size_t i = 0; i < n[0]; i += buffer_size)
				{
                    // get random numbers
					rng[thread_id]->next_float(buffer, buffer_size);

					const std::size_t ii_max = std::min(buffer_size, n[0] - i);
					for (std::size_t ii = 0; ii < ii_max; ++ii)
					{
                        // if the random number is larger than 0.5, make the spin point up (0x1),
                        // and otherwise make it point down (0x0)
						lattice[j * n[0] + (i + ii)] = (buffer[ii] > 0.5F ? 0x1 : 0x0);
					}
				}
			}
		}
	}

    //! \brief Destructor
    spin_system<2>::~spin_system()
	{
        // delete all random number generators
        const std::size_t num_threads = omp_get_max_threads();
		for (std::size_t i = 0; i < num_threads; ++i)
		{
			if (rng[i] != nullptr)
			{
				delete rng[i];
				rng[i] = nullptr;
			}
		}
        rng.clear();
	}

    //! \brief Lattice sweep (update the entire lattice)
    //!
    //! \param beta inverse temperature to be used for the sweep
	void spin_system<2>::update(const float beta)
	{
        // probability for adding aligned neighboring sites to the cluster
		const float p_add = 1.0F - static_cast<float>(std::exp(-2.0 * beta));

        // labeling within tiles
		assign_labels(p_add);

        // merging the tiles by establishing label equivalences
		merge_labels(p_add);

        // resolve all labels for the final labeling
		resolve_labels();

        // flip clusters
		flip_spins();

        // the lattice has changes: internal energy and magnetization have to be
        // measured when calling get_energy() and get_magnetization() the next time
		call_measurement_routine = true;
	}

    //! \brief Determine the internal energy of the system
    //!
    //! \return negative internal energy per sit
	double spin_system<2>::get_energy()
	{
		if (call_measurement_routine)
		{
			measure();
		}

		return energy;
	}

    //! \brief Determine the magnetization of the system
    //!
    //! \return magnetization per site
	double spin_system<2>::get_magnetization()
	{
		if (call_measurement_routine)
		{
			measure();
		}

		return magnetization;
	}

    //! \brief Loop over all tiles of the lattice and apply e.g. ccl_selflabeling
    //!
    //! \param p_add probability for adding aligned nearest neighbor sites to a cluster
	void spin_system<2>::assign_labels(const float p_add)
	{
		#pragma omp parallel for collapse(2)
		for (std::size_t j = 0; j < n[1]; j += chunk[1])
		{
			for (std::size_t i = 0; i < n[0]; i += chunk[0])
			{
				const std::size_t n_offset[2] = {i, j};
				const std::size_t n_sub[2] = {std::min(chunk[0], n[0] - i), std::min(chunk[1], n[1] - j)};

				if (n_sub[0] == chunk[0])
				{
                    // we can call a version of that method with the extent in 0-direction being
                    // a compile time constant (hopefully allowing the compiler to do better optimizations)
					ccl_selflabeling<chunk[0]>(p_add, n_offset, n_sub);
				}
				else
				{
					ccl_selflabeling(p_add, n_offset, n_sub);
				}
			}
		}
	}

    //! \brief Connect all tiles
    //!
    //! \param p_add probability for adding aligned nearest neighbor sites to the cluster
	void spin_system<2>::merge_labels(const float p_add)
	{
		#pragma omp parallel
		{
			const std::uint32_t thread_id = omp_get_thread_num();
			constexpr std::size_t buffer_size = chunk[0] + chunk[1];
			float buffer[buffer_size];

			#pragma omp for collapse(2)
			for (std::size_t j = 0; j < n[1]; j += chunk[1])
			{
				for (std::size_t i = 0; i < n[0]; i += chunk[0])
				{
					rng[thread_id]->next_float(buffer, buffer_size);

					const std::size_t jj_max = std::min(chunk[1], n[1] - j);
					const std::size_t ii_max = std::min(chunk[0], n[0] - i);

                    // merge in 1-direction
					for (std::size_t ii = 0; ii < ii_max; ++ii)
					{
						const std::size_t idx_0 = (j + jj_max - 1) * n[0] + (i + ii);
						const std::size_t idx_1 = ((j + jj_max) % n[1]) * n[0] + (i + ii);

						if (buffer[ii] < p_add && lattice[idx_0] == lattice[idx_1])
						{
							std::uint32_t a = cluster[idx_0];
							std::uint32_t b = cluster[idx_1];
							if (a != b)
							{
								merge(&cluster[0], a, b);
							}
						}
					}

                    // merge in 0-direction
					for (std::size_t jj = 0; jj < jj_max; ++jj)
					{
						const std::size_t idx_0 = (j + jj) * n[0] + (i + ii_max - 1);
						const std::size_t idx_1 = (j + jj) * n[0] + (i + ii_max) % n[0];

						if (buffer[chunk[0] + jj] < p_add && lattice[idx_0] == lattice[idx_1])
						{
							std::uint32_t a = cluster[idx_0];
							std::uint32_t b = cluster[idx_1];
							if (a != b)
							{
								merge(&cluster[0], a, b);
							}
						}
					}
				}
			}
		}
	}

    //! \brief Helper method to establish label equivalences, thus merging clusters
    //!
	//! Important: the whole procedure works only because of 1) we start with all initial labels are
	//! given by the 1-D index with ptr[X] = X (each site is its own cluster and root),
	//! and 2) we always replace by the minimum when establishing the equivalence of two labels!
	//!
    //! Given an assumed label equivalence (here A and B are the same), this method alters the labels
    //! pointed to by ptr such that afterwards ptr[A] and ptr[B] contain the min(A, B).
    //! If in the mean time other threads have applied label equivalences, it might be possible that
    //! any of ptr[A] and ptr[B] contains a value lower than min(A, B), e.g. if label C is equivalen to A
    //! and C < A, then ptr[A] would hold the value C. In the mean time another thread could have
    //! established ptr[C] = D, and so on.
    //! However, in any case following these label equivalences down to the root (X = ptr[X]) gives the
    //! final label for the clusters.
    //! \n
    //! The atomic_min() method makes sure that the field behind ptr is not corrupted when establishing
    //! label equivalences in a multi-threaded context.
    //!
    //! \param ptr pointer to the field holding all cluster labels (and equivalences)
    //! \param a cluster label
    //! \param b cluster label (there is an equivalenc of a and b that needs to be established)
    void spin_system<2>::merge(std::uint32_t* ptr, std::uint32_t a, std::uint32_t b)
    {
        // the loop is guaranteed to break somewhen (maybe not that obvious)
        while (true)
        {
            // c holds either the old value pointed to by ptr[b] in case of A is smaller,
            // or the actual minimum if the assumption that A is smaller is wrong
            std::uint32_t c = atomic_min(&ptr[b], a);

            // successfully established the equivalence of A and B!
            // in the first loop iteration it might be that C != A, but if in the meantime no
            // other thread changes the equivalence, then ptr[B] = A and C will be equal to A
            if (c == a)
            {
                break;
            }

            // the assumption that A is smaller is true and we successfully established the
            // equivalence of A and B. We now have to adapt the already existing equivalence of
            // B and C (because previously ptr[B] held C). If C != B, we now have to establish
            // the equivalence of A and C, which is like calling this routine with B = C.
            if (c > a)
            {
                b = c;
            }

            // the assumption that A is smaller is false and C (!= B) is smaller than A.
            // we now have to establish the equivalence of A and C where not it is assumed that
            // C is the smaller one, which might have changed in the meantime.
            if (c < a)
            {
                b = a;
                a = c;
            }
        }
    }

    //! \brief Replace the data pointed to by ptr by desired if and only if desired is smaller
    //!
    //! \param ptr pointer to the data to be replaced by desired if and only if desired is smaller
    //! \param desired value to replace the data pointed to by ptr if desired is smaller
    //! \return value pointed to by ptr (the former value pointed to by ptr, if desired is smaller)
    std::uint32_t spin_system<2>::atomic_min(volatile std::uint32_t* ptr, const std::uint32_t desired)
    {
        while (true)
        {
            // backup the value pointed to by ptr
            std::uint32_t old = *ptr;

            // if this value is already lower than the assumed (desired) one, return it...
            if (old <= desired)
            {
                return *ptr;
            }

            // ...if not, try to replace it by the desired one.
            // The replacement is successful only if in the meantime the value pointed to by ptr
            // did not change. If so (__sync_bool...() returns true), return the old value pointed to by ptr.
            // Otherwise, try again: it might be possible that the new value pointed to by ptr is still
            // larger than the desired one.
            if (__sync_bool_compare_and_swap(ptr, old, desired))
            {
                return old;
            }
        }
    }

    //! \brief Resolve all label equivalences
    //!
    //! Just go through the entire lattice and for each site follow the chain of label equivalences
    //! down to the root (X = ptr[X]).
    //! Then establish the link
	void spin_system<2>::resolve_labels()
	{
		#pragma omp parallel for
		for (std::size_t i = 0; i < (n[0] * n[1]); ++i)
		{
			std::uint32_t c = cluster[i];
			while (c != cluster[c])
			{
				c = cluster[c];
			}
			cluster[i] = c;
		}
	}

    //! \brief Flip clusters
    //!
    //! Clusters are flipped as a whole with probability 0.5.
    //! As we use the 1-D index for the initial label assignment, the probability for the root label X
    //! of each cluster to be either an even or an odd number is the same.
    //! We thus flip a cluster only if X is odd, that is, if (X & 0x1) is equal to 0x1.
    //! Flipping is implemented via a bitwise XOR operation.
	void spin_system<2>::flip_spins()
	{
		#pragma omp parallel for
		for (std::size_t i = 0; i < (n[0] * n[1]); ++i)
		{
			lattice[i] ^= (cluster[i] & 0x1);
		}
	}

    //! \brief Determine the internal energy and magnetization, both per site
    //!
    //! Spins are have values -1 (~ 0x0) and +1 (~ 0x1)
    //! \n
    //! Energy = \sum_i e_i with e_i = \sum_<i,j> l_i * l_j
    //! Magnetization = \sum_i l_i
	void spin_system<2>::measure()
	{
		std::int64_t i_energy = 0L;
		std::int64_t i_magnetization = 0L;

		#pragma omp parallel for reduction(+ : i_energy, i_magnetization)
		for (std::size_t j = 0; j < n[1]; ++j)
		{
			for (std::size_t i = 0; i < n[0]; ++i)
				{
                    // l_{0,1,2} or either 0x0 or 0x1
					const std::int32_t l_0 = static_cast<std::int32_t>(lattice[j * n[0] + i]);
					const std::int32_t l_1 = static_cast<std::int32_t>(lattice[j * n[0] + (i + 1) % n[0]]);
					const std::int32_t l_2 = static_cast<std::int32_t>(lattice[((j + 1) % n[1]) * n[0] + i]);

					// l_0  l_1  l_2  i_energy  e_i
					// 0x0  0x0  0x0  0x0		+2
					// 0x0  0x1  0x0  0x1        0
					// 0x0  0x0  0x1  0x1        0
					// 0x0  0x1  0x1  0x2       -2
					// 0x1  0x0  0x0  0x2       -2
					// 0x1  0x1  0x0  0x1        0
					// 0x1  0x0  0x1  0x1        0
					// 0x1  0x1  0x1  0x0       +2
					i_energy += ((l_0 ^ l_1) + (l_0 ^ l_2));
					i_magnetization += l_0;

					// mapping:
					// x = (l_0 ^ l_1) + (l_0 ^ l_2)
					// -e_i = 2 * x - 2
				}
		}

        const std::int64_t volume = n[0] * n[1];
		// negative internal energy: this comprises all lattice sites
		energy = static_cast<double>(2 * i_energy - 2 * volume) / volume;
		// magnetization
		magnetization = static_cast<double>(2 * i_magnetization - volume) / volume;

		// measurement done: we can call get_energy() and get_magnetization() now until the next sweep
		call_measurement_routine = false;
	}
}
