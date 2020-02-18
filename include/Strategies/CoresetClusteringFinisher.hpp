#pragma once

#include <numeric>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
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

    virtual precision finishClustering(KmeansData<precision, int_size>* const kmeansData) = 0;
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

    precision finishClustering(KmeansData<precision, int_size>* const kmeansData) override;
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

    precision finishClustering(KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
precision SharedMemoryCoresetClusteringFinisher<precision, int_size>::finishClustering(
  KmeansData<precision, int_size>* const kmeansData)
{
    this->pUpdater->findAndUpdateClosestClusters(kmeansData);

    return std::accumulate(kmeansData->sqDistancesBegin(), kmeansData->sqDistancesEnd(), 0.0);
}

template <typename precision, typename int_size>
precision MPICoresetClusteringFinisher<precision, int_size>::finishClustering(
  KmeansData<precision, int_size>* const kmeansData)
{
    this->pUpdater->findAndUpdateClosestClusters(kmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansData->myLength(), MPI_INT, kmeansData->clusteringData(),
                   kmeansData->lengthsData(), kmeansData->displacementsData(), MPI_INT, MPI_COMM_WORLD);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansData->myLength(), mpi_type_t, kmeansData->sqDistancesData(),
                   kmeansData->lengthsData(), kmeansData->displacementsData(), mpi_type_t, MPI_COMM_WORLD);

    return std::accumulate(kmeansData->sqDistancesBegin(), kmeansData->sqDistancesEnd(), 0.0);
}
}  // namespace HPKmeans