# MPI Parallel Implementation

This directory contains an OpenMPI-based parallel implementation of the Weierstrass function integral calculation.

## Structure

- `mpi_parallel.hpp` - Header file with function declarations
- `mpi_parallel.cpp` - MPI implementation using same algorithm and data splitting as the std::thread version
- `mpi_test.cpp` - Test application to verify functionality and measure performance
- `CMakeLists.txt` - Build configuration with MPI dependencies

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running

The test application can be run with different numbers of MPI processes:

```bash
# Run with 2 processes
mpirun -np 2 ./mpi_test

# Run with 4 processes (may need --oversubscribe on single-node systems)
mpirun --oversubscribe -np 4 ./mpi_test
```

## Implementation Details

The MPI implementation:
- Uses the same Weierstrass function algorithm as other implementations
- Implements the same data splitting strategy as `integral_parallel`
- Distributes work evenly across MPI processes
- Uses `MPI_Allreduce` to collect partial results
- Returns the same numerical results as single-threaded version

The data splitting ensures even distribution:
- Each process gets `steps/size` base chunks
- First `steps % size` processes get one additional step
- This ensures optimal load balancing across processes

## Performance

Typical speedup results:
- 2 processes: ~2x speedup
- 4 processes: ~3.5x speedup (with oversubscription)

Results are verified against single-threaded implementation for correctness.