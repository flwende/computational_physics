# Computational Physics
This repository contains a collection of CP projects

# Ising
Directory `ising` contains an implementation of the Swendsen Wang (multi-)cluster algorithm for the 2-dimensional Ising model.

### Build
Create directories `bin` and `obj` and run `make` (maybe some modifications to the Makefile are needed).
You can adjust the number of lattice update in the `src/ising_2d.cpp` file (see the head of that file).

### Run
```
$> export OMP_NUM_THREADS=32
$> ./bin/ising_2d.x 128 128
```

### Output (sample)
The executable has been created with the given Makefile and g++-7.3 (CentOS 7.3).
The execution happened on a dual socket Intel Xeon E5-2630v3 compute node with 16 CPU cores (+HT) using 32 OpenMP threads.
```
update time per site (lattice = 128 x 128): 8.17733 ns
internal energy per site: -1.41848
absolute magnetization per site: 0.550003
```
