# Computational Physics
This repository contains a collection of CP projects.

### Build
Type the below commands to compile any of the below listed programs (`--target`) using CMake:
```
$> mkdir build && cd build
$> cmake ..
$> cmake --build . --target <PROGRAM>
$> make
```

# Random
Directory `random` contains an implementation of a linear congruential generator (lcg) with (optional) random state shuffling to improve the quality of the generated random numbers.

### Build (just for testing purposes)
See above build section:
```
$> cmake --build . --target TestRNG && make
```

### Run
```
$> export OMP_NUM_THREADS=32
$> ./bin/TestRNG
```

### Output (sample)
Example output with 32 OpenMP threads:
```
...
thread 17: 
0.00492056 0.838682 0.637546 0.971339 0.625987 0.955005 0.67618 0.311519 0.976153 0.541141 0.906482 0.359729 0.696381 0.0250481 0.526235 0.714885 0.551438 0.909821 0.924546 0.985797 0.676524 0.874883 0.333021 0.436596 5.23992e-05 0.0423673 0.909534 0.608339 0.196104 0.170222 0.595131 0.729294 
time per random number = 0.0491182 ns
mrd. random numbers per second = 20.359
```

# Ising
Directory `ising` contains an implementation of the Swendsen Wang (multi-)cluster algorithm for the 2-dimensional Ising model.

### Build
See above build section:
```
$> cmake --build . --target Ising2D && make
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
