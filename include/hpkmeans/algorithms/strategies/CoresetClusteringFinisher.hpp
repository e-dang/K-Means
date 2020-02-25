#pragma once

#include <mpi.h>

#include <hpkmeans/algorithms/strategies/ClosestClusterUpdater.hpp>
#include <hpkmeans/data_types/kmeans_state.hpp>
#include <hpkmeans/utils/mpi_class.hpp>
#include <numeric>

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractCoresetClusteringFinisher
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> pUpdater;

public:
    AbstractCoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) : pUpdater(updater)
    {
    }

    virtual ~AbstractCoresetClusteringFinisher() = default;

    virtual void finishClustering(KmeansState<precision, int_size>* const kmeansState) = 0;
};

template <typename precision, typename int_size>
class SharedMemoryCoresetClusteringFinisher : public AbstractCoresetClusteringFinisher<precision, int_size>
{
public:
    SharedMemoryCoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractCoresetClusteringFinisher<precision, int_size>(updater)
    {
    }

    ~SharedMemoryCoresetClusteringFinisher() = default;

    void finishClustering(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class MPICoresetClusteringFinisher :
    public AbstractCoresetClusteringFinisher<precision, int_size>,
    public MPIClass<precision, int_size>
{
private:
    using MPIClass<precision, int_size>::mpi_precision;
    using MPIClass<precision, int_size>::mpi_int_size;

public:
    MPICoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractCoresetClusteringFinisher<precision, int_size>(updater)
    {
    }

    ~MPICoresetClusteringFinisher() = default;

    void finishClustering(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
void SharedMemoryCoresetClusteringFinisher<precision, int_size>::finishClustering(
  KmeansState<precision, int_size>* const kmeansState)
{
    this->pUpdater->findAndUpdateClosestClusters(kmeansState);

    // return std::accumulate(kmeansState->sqDistancesBegin(), kmeansState->sqDistancesEnd(), 0.0);
}

template <typename precision, typename int_size>
void MPICoresetClusteringFinisher<precision, int_size>::finishClustering(
  KmeansState<precision, int_size>* const kmeansState)
{
    this->pUpdater->findAndUpdateClosestClusters(kmeansState);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansState->myLength(), mpi_int_size, kmeansState->clusteringData(),
                   kmeansState->lengthsData(), kmeansState->displacementsData(), mpi_int_size, MPI_COMM_WORLD);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansState->myLength(), mpi_precision, kmeansState->sqDistancesData(),
                   kmeansState->lengthsData(), kmeansState->displacementsData(), mpi_precision, MPI_COMM_WORLD);

    // return std::accumulate(kmeansState->sqDistancesBegin(), kmeansState->sqDistancesEnd(), 0.0);
}
}  // namespace HPKmeans