#pragma once

#include <chrono>

#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "boost/generator_iterator.hpp"
#include "boost/random.hpp"
#include "mpi.h"

typedef boost::mt19937 RNGType;

inline int64_t getTime()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline double getRandDouble01()
{
    thread_local static RNGType rng(getTime());
    thread_local static boost::uniform_real<double> dblRange(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<double>> dblDistr(rng, dblRange);

    return dblDistr();
}

inline double getRandDouble01MPI()
{
    static int rank = -1;
    if (rank == -1) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    thread_local static RNGType rng(getTime() * (rank + 10));
    thread_local static boost::uniform_real<double> dblRange(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<double>> dblDistr(rng, dblRange);

    return dblDistr();
}

inline MPIData getMPIData(const int_fast32_t& totalNumData)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // number of datapoints allocated for each process to compute
    int_fast32_t chunk = totalNumData / numProcs;
    int_fast32_t scrap = chunk + (totalNumData % numProcs);

    std::vector<int_fast32_t> lengths(numProcs);        // size of each sub-array to gather
    std::vector<int_fast32_t> displacements(numProcs);  // index of each sub-array to gather
    for (int i = 0; i < numProcs; i++)
    {
        lengths.at(i)       = chunk;
        displacements.at(i) = i * chunk;
    }
    lengths.at(numProcs - 1) = scrap;

    return MPIData{ rank, numProcs, lengths, displacements };
}

inline int_fast32_t getTotalNumDataMPI(const Matrix* const data)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int_fast32_t totalNumData = 0;
    int_fast32_t localNumData = data->getNumData();

    MPI_Allreduce(&localNumData, &totalNumData, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    return totalNumData;
}