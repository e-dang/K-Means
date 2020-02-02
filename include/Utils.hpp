#pragma once

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