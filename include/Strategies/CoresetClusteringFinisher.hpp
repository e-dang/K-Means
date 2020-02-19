#pragma once

#include <numeric>

#include "Containers/Definitions.hpp"
#include "Containers/KmeansState.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"
#include "mpi.h"
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

    virtual precision finishClustering(KmeansState<precision, int_size>* const kmeansState) = 0;
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

    precision finishClustering(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class MPICoresetClusteringFinisher : public AbstractCoresetClusteringFinisher<precision, int_size>
{
public:
    MPICoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractCoresetClusteringFinisher<precision, int_size>(updater)
    {
    }

    ~MPICoresetClusteringFinisher() = default;

    precision finishClustering(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
precision SharedMemoryCoresetClusteringFinisher<precision, int_size>::finishClustering(
  KmeansState<precision, int_size>* const kmeansState)
{
    this->pUpdater->findAndUpdateClosestClusters(kmeansState);

    return std::accumulate(kmeansState->sqDistancesBegin(), kmeansState->sqDistancesEnd(), 0.0);
}

template <typename precision, typename int_size>
precision MPICoresetClusteringFinisher<precision, int_size>::finishClustering(
  KmeansState<precision, int_size>* const kmeansState)
{
    this->pUpdater->findAndUpdateClosestClusters(kmeansState);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansState->myLength(), MPI_INT, kmeansState->clusteringData(),
                   kmeansState->lengthsData(), kmeansState->displacementsData(), MPI_INT, MPI_COMM_WORLD);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansState->myLength(), mpi_type_t, kmeansState->sqDistancesData(),
                   kmeansState->lengthsData(), kmeansState->displacementsData(), mpi_type_t, MPI_COMM_WORLD);

    return std::accumulate(kmeansState->sqDistancesBegin(), kmeansState->sqDistancesEnd(), 0.0);
}
}  // namespace HPKmeans