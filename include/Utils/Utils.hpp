#pragma once

#include <mpi.h>

#include <chrono>

#include "Containers/DataClasses.hpp"
#include "boost/generator_iterator.hpp"
#include "boost/random.hpp"

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

template <typename int_size>
inline MPIData<int_size> getMPIData(const int_size& totalNumData)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // number of datapoints allocated for each process to compute
    int_size chunk = totalNumData / numProcs;
    int_size scrap = chunk + (totalNumData % numProcs);

    std::vector<int_size> lengths(numProcs);        // size of each sub-array to gather
    std::vector<int_size> displacements(numProcs);  // index of each sub-array to gather
    for (int i = 0; i < numProcs; ++i)
    {
        lengths.at(i)       = chunk;
        displacements.at(i) = i * chunk;
    }
    lengths.at(numProcs - 1) = scrap;

    return MPIData<int_size>(rank, numProcs, lengths, displacements);
}

template <typename precision, typename int_size>
inline int_size getTotalNumDataMPI(const Matrix<precision, int_size>* const data)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int_size totalNumData;
    int_size localNumData = data->rows();

    MPI_Allreduce(&localNumData, &totalNumData, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return totalNumData;
}
}  // namespace HPKmeans