#pragma once

#include <mpi.h>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <chrono>

typedef boost::mt19937 RNGType;

namespace hpkmeans
{
inline int64_t getTime()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline double getRandFraction()
{
    // thread_local static RNGType rng(getTime());
    thread_local static RNGType rng(0);
    thread_local static boost::uniform_real<double> distr(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<double>> gen(rng, distr);

    return gen();
}

inline int getCommRank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

inline int getCommSize()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

template <typename T>
MPI_Datatype matchMPIType()
{
    MPI_Datatype dtype;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(T), &dtype);
    return dtype;
}

template <typename Iter>
void print(Iter begin, Iter end)
{
    for (; begin != end; ++begin)
    {
        std::cout << *begin << " ";
    }
    std::cout << '\n' << std::flush;
}
}  // namespace hpkmeans