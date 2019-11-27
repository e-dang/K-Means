# K-Means

# Under Development!!!!

### Compiling
In the top level directory of the project, run the following commands:
```
mkdir build
cd build
cmake ..
make
```

### Executing with MPI
```
$ mpirun -np "NUM_PROCS" ./kmeans

Where NUM_PROCS is an integer.
```

### Dependencies
- [OpenMP](https://www.openmp.org/)
- [Boost](https://www.boost.org/)
- [CMake](https://cmake.org/) 3.0
- [OpenMPI](https://www.open-mpi.org/)
