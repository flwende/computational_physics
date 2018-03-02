// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(SPIN_SYSTEM_HPP)
#define SPIN_SYSTEM_HPP

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <simd/simd.hpp>
#include <random/random.hpp>

#if !defined(XXX_NAMESPACE)
#define XXX_NAMESPACE fw
#endif

namespace XXX_NAMESPACE
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Class implementing the Swendsen Wang cluster algorithm for the D-dimensional Ising model
	//!
	//! \tparam D dimension
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <std::size_t D>
	class spin_system;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//! \brief Class implementing the Swendsen Wang cluster algorithm for the 2-dimensional Ising model
	//!
	//! Reference: R. H. Swendsen and J. S. Wang,
	//!            "Nonuniversal critical dynamics in Monte Carlo simulations"
	//!            Phys. Rev. Lett., 58:86-88, Jan 1987
	//!
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <>
	class spin_system<2>
	{
		//! Tile size for parallel processing: the innermost dimension should be a multiple of the SIMD width
		//! of the target platform
		static constexpr std::size_t chunk[2] = {simd::type<std::uint32_t>::width, 8};

		//! Lattice extent
		const std::size_t n[2];
		//! Lattice: we just need to store 0 or 1 (spin down or up orientation)
		std::vector<std::uint8_t> lattice;
		//! Cluster: the largest possible label is 0xFFFFFFFF
		std::vector<std::uint32_t> cluster;
		//! Random number generator: if you use the lcg32 generator, make sure you are
		//! compiling with RANDOM_SHUFFLE_STATE (otherwise, random numbers are too bad)
		std::vector<random*> rng;

		//! Energy per site
		double energy;
		//! Magnetization per site
		double magnetization;
		//! Do we have to call measure()?
		bool call_measurement_routine;

	public:

		//! \brief Constructor
		//!
		//! \param n lattice extent
		spin_system(const std::size_t (&n) [2]);

		//! \brief Destructor
		~spin_system();

		//! \brief Lattice sweep (update the entire lattice)
		//!
		//! Each update comprises calling the following methods
		//! \n\n
		//! 1. assign_labels (using e.g. ccl_selflabeling)
		//! \n
		//! 2. merge_labels
		//! \n
		//! 3. resolve_labels
		//! \n
		//! 4. flip_spins
		//! \n\n
		//! Steps 1-3: find all clusters (connected components)
		//! \n
		//! Sep 4: flip clusters individually
		//!
		//! \param beta inverse temperature to be used for the sweep
		void update(const float beta);

		//! \brief Determine the internal energy of the system
		//!
		//! \return negative internal energy per site
		double get_energy();

		//! \brief Determine the magnetization of the system
		//!
		//! \return magnetization per site
		double get_magnetization();
		
	private:

		//! \brief Loop over all tiles of the lattice and apply e.g. ccl_selflabeling
		//!
		//! \param p_add probability for adding aligned nearest neighbor sites to a cluster
		void assign_labels(const float p_add);

		//! \brief Connected component labeling (ccl) based on an idea of Coddington and Baillie within tiles
		//!
		//! \tparam N_0 tile size in 0-direction
		//! \param p_add probability for adding aligned nearest neighbor sites to the cluster
		//! \param n_offset tile offset w.r.t. to (0, 0)
		//! \param n_sub extent of the tile
		template <std::size_t N_0 = 0>
		void ccl_selflabeling(const float p_add, const std::size_t (&n_offset) [2], const std::size_t (&n_sub) [2]);

		//! \brief Connect all tiles
		//!
		//! \param p_add probability for adding aligned nearest neighbor sites to the cluster
		void merge_labels(const float p_add);

		//! \brief Helper method to establish label equivalences, thus merging clusters
		//!
		//! \param ptr pointer to the field holding all cluster labels (and equivalences)
		//! \param a cluster label
		//! \param b cluster label (there is an equivalenc of a and b that needs to be established)
		void merge(std::uint32_t* ptr, std::uint32_t a, std::uint32_t b);

		//! \brief Replace the data pointed to by ptr by desired if and only if desired is smaller
		//!
		//! \param ptr pointer to the data to be replaced by desired if and only if desired is smaller
		//! \param desired value to replace the data pointed to by ptr if desired is smaller
		//! \return value pointed to by ptr (the former value pointed to by ptr, if desired is smaller)
		std::uint32_t atomic_min(volatile std::uint32_t* ptr, const std::uint32_t desired);

		//! \brief Resolve all label equivalences
		void resolve_labels();

		//! \brief Flip clusters
		void flip_spins();

		//! \brief Determine the internal energy and magnetization, both per site
		void measure();
	};
}

#undef XXX_NAMESPACE

#include <spin_system_selflabeling.hpp>

#endif
