// Copyright (c) 2017-2018 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <omp.h>
#include <spin_system.hpp>

static constexpr std::size_t n_0_default = 32;
static constexpr std::size_t n_1_default = 32;
static constexpr float temperature_default =  2.2691853142130221F;

static constexpr std::size_t n_warmup = 10000;
static constexpr std::size_t n_sep = 20;
static constexpr std::size_t n_measurement = (n_sep * 10000);

int main(int argc, char **argv)
{
    // get lattice extent from command line (if there are any arguments)
	const std::size_t n_0 = (argc > 1 ? atoi(argv[1]) : n_0_default);
	const std::size_t n_1 = (argc > 2 ? atoi(argv[2]) : n_1_default);
	const std::size_t n[2] = {n_0, n_1};

    // temperature to simulate at: default is the critical temperatur for the 2-D Ising model
	const float temperature = (argc > 3 ? atof(argv[3]) : temperature_default);

    // create spin system
    fw::spin_system<2> s(n);

    // thermalization
	for (std::size_t i = 0; i < n_warmup; ++i)
    {
		s.update(1.0F / temperature);
    }

    // measurement
	double energy = 0.0;
	double magnetization = 0.0;
    std::size_t num_measurements = 0;

	double time = omp_get_wtime();
	for (std::size_t i = 0; i < n_measurement; ++i)
    {
        s.update(1.0F / temperature);

        // take measurements every n_sep update
        if (i > 0 && (i % n_sep) == 0)
        {
            energy += s.get_energy();
            magnetization += std::abs(s.get_magnetization());
            ++num_measurements;
        }
    }
	time = omp_get_wtime() - time;

    // output the update time per site
    std::cout << "update time per site (lattice = " << n[0] << " x " << n[1] << "): ";
	std::cout << time * 1.0E9 / (static_cast<std::size_t>(n_measurement) * n[0] * n[1]) << " ns" << std::endl;

    // and mean internal energy and magnetization, both per site
	energy /= num_measurements;
    std::cout << "internal energy per site: " << energy << std::endl;
	magnetization /= num_measurements;
    std::cout << "absolute magnetization per site: " << magnetization << std::endl;

	return 0;
}
