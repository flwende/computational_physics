# Computational Physics
This repository contains a collection of CP projects.

### Build
Type the below commands to compile any of the below listed programs (`--target`) using CMake:
```
$> mkdir build && cd build
$> cmake ..
$> cmake --build . [--target <PROGRAM>]
```

# Random
Directory `random` contains an implementation of a linear congruential generator (lcg32) with (optional) random state shuffling to improve the quality of the generated random numbers.

### Build (just for testing purposes)
See above build section:
```
$> cmake --build . --target TestRNG
$> ./bin/TestRNG -h
..
Options:
  --help, -h      Show this help message
  --rng=<name>    Set RNG type (default: lcg32)
  --target=<name> Set target device (default: cpu)
      Supported target devices: cpu, amd_gpu
  --id=<number>   Set reporting ID (default: 0)
```

### Run
The test program reads the environment variable `NUM_THREADS`. If compiled with Clang and HIP support, `--target=amd_gpu` can be specified and the meaning of `NUM_THREADS` is multi-processors then, e.g.,
```
$> export NUM_THREADS=16
$> ./bin/TestRNG --rng=lcg32 --target=amd_gpu --id=2
```

### Output (sample)
Example output with 16 multi-processors (`NUM_THREADS=16`) and `--target=amd_gpu`:
```
0.932042 0.975611 0.0124143 0.145947 0.0482002 0.768371 0.226047 0.0643613 0.530811 0.275953 0.0698682 0.220406 0.552128 0.908475 0.143886 0.804557 0.133395 0.897812 0.743874 0.180452 0.794722 0.901898 0.207154 0.0259293 0.616612 0.702956 0.165535 0.126219 0.493634 0.283856 0.547699 0.812443 0.282718 0.504096 0.583031 0.290856 0.948764 0.334882 0.156375 0.252362 0.234238 0.350846 0.0859821 0.393544 0.22366 0.697054 0.506967 0.9743 0.417221 0.602808 0.973231 0.654051 0.284861 0.252255 0.131135 0.245945 0.600849 0.132887 0.776732 0.498785 0.237845 0.988583 0.561753 0.272553
Time per random number = 0.067842 ns
Billion random numbers per second = 14.7401
```

# Ising
Directory `ising` contains an implementation of the Swendsen Wang (multi-)cluster algorithm for the 2-dimensional Ising model.

### Build
See above build section:
```
$> cmake --build . --target Ising2D
```
You can adjust the number of lattice update in the `src/ising_2d.cpp` file (see the head of that file).

### Run
```
$> export OMP_NUM_THREADS=32
$> ./bin/Ising2D 128 128
```

### Output (sample)
Example output with 32 OpenMP threads:
```
update time per site (lattice = 128 x 128): 8.17733 ns
internal energy per site: -1.41848
absolute magnetization per site: 0.550003
```
*NOTE*: this is a Monte Carlo simulation. Changing thread counts and/or number of lattice updates can result in different simulation outputs. However, for the same configuration, the output should be reproducibly the same. Testing for "correctness" can be also done by comparing against exact calculations (see `ising/verification/info.pdf`). For the critical 2-dimensional Ising model with 128 x 128 lattice, for instance, the negative internal energy per spin is `-1.419076272084983`. The value above (`-1.41848`) is a Monte Carlo approximation of this. Running the simulation for longer times should give better results.
