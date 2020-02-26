#pragma once

#include <mpi.h>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <chrono>

typedef boost::mt19937 RNGType;
namespace HPKmeans
{
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
    if (rank == -1)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    thread_local static RNGType rng(getTime() * (rank + 10));
    thread_local static boost::uniform_real<double> dblRange(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<double>> dblDistr(rng, dblRange);

    return dblDistr();
}
}  // namespace HPKmeans