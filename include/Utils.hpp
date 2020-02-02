#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "boost/generator_iterator.hpp"
#include "boost/random.hpp"
#include "mpi.h"

typedef boost::mt19937 RNGType;

inline float getRandFloat01()
{
    thread_local static RNGType rng(time(NULL));
    thread_local static boost::uniform_real<> floatRange(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    return floatDistr();
}

inline float getRandFloat01MPI()
{
    static int rank = -1;
    if (rank == -1) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    thread_local static RNGType rng(time(NULL) * (rank + 1));
    thread_local static boost::uniform_real<> floatRange(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    return floatDistr();
}

inline MPIData getMPIData(const int& totalNumData)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // number of datapoints allocated for each process to compute
    int chunk = totalNumData / numProcs;
    int scrap = chunk + (totalNumData % numProcs);

    std::vector<int> lengths(numProcs);        // size of each sub-array to gather
    std::vector<int> displacements(numProcs);  // index of each sub-array to gather
    for (int i = 0; i < numProcs; i++)
    {
        lengths[i]       = chunk;
        displacements[i] = i * chunk;
    }
    lengths[numProcs - 1] = scrap;

    return MPIData{ rank, numProcs, lengths, displacements };
}