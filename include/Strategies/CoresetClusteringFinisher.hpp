#pragma once

#include <numeric>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"
#include "mpi.h"
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class AbstractCoresetClusteringFinisher
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> pUpdater;

public:
    AbstractCoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        pUpdater(updater){};

    virtual ~AbstractCoresetClusteringFinisher() = default;

    virtual precision finishClustering(KmeansData<precision, int_size>* const kmeansData) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SharedMemoryCoresetClusteringFinisher : public AbstractCoresetClusteringFinisher<precision, int_size>
{
public:
    SharedMemoryCoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractCoresetClusteringFinisher<precision, int_size>(updater){};

    ~SharedMemoryCoresetClusteringFinisher() = default;

    precision finishClustering(KmeansData<precision, int_size>* const kmeansData) override
    {
        this->pUpdater->findAndUpdateClosestClusters(kmeansData);

        return std::accumulate(kmeansData->sqDistances->begin(), kmeansData->sqDistances->end(), 0.0);
    }
};

template <typename precision = double, typename int_size = int32_t>
class MPICoresetClusteringFinisher : public AbstractCoresetClusteringFinisher<precision, int_size>
{
public:
    MPICoresetClusteringFinisher(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractCoresetClusteringFinisher<precision, int_size>(updater){};

    ~MPICoresetClusteringFinisher() = default;

    precision finishClustering(KmeansData<precision, int_size>* const kmeansData) override
    {
        this->pUpdater->findAndUpdateClosestClusters(kmeansData);

        MPI_Allgatherv(MPI_IN_PLACE, kmeansData->lengths.at(kmeansData->rank), MPI_INT, kmeansData->clustering->data(),
                       kmeansData->lengths.data(), kmeansData->displacements.data(), MPI_INT, MPI_COMM_WORLD);

        MPI_Allgatherv(MPI_IN_PLACE, kmeansData->lengths.at(kmeansData->rank), mpi_type_t,
                       kmeansData->sqDistances->data(), kmeansData->lengths.data(), kmeansData->displacements.data(),
                       mpi_type_t, MPI_COMM_WORLD);

        return std::accumulate(kmeansData->sqDistances->begin(), kmeansData->sqDistances->end(), 0.0);
    }
};
}  // namespace HPKmeans