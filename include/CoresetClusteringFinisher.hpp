#pragma once

#include <numeric>

#include "ClosestClusterUpdater.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "mpi.h"

class AbstractCoresetClusteringFinisher
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater> pUpdater;

public:
    AbstractCoresetClusteringFinisher(AbstractClosestClusterUpdater* updater) : pUpdater(updater){};

    virtual ~AbstractCoresetClusteringFinisher(){};

    virtual value_t finishClustering(KmeansData* const kmeansData) = 0;
};

class SharedMemoryCoresetClusteringFinisher : public AbstractCoresetClusteringFinisher
{
public:
    SharedMemoryCoresetClusteringFinisher(AbstractClosestClusterUpdater* updater) :
        AbstractCoresetClusteringFinisher(updater){};

    virtual ~SharedMemoryCoresetClusteringFinisher(){};

    value_t finishClustering(KmeansData* const kmeansData) override
    {
        pUpdater->findAndUpdateClosestClusters(kmeansData);

        return std::accumulate(kmeansData->pSqDistances->begin(), kmeansData->pSqDistances->end(), 0.0);
    }
};

class MPICoresetClusteringFinisher : public AbstractCoresetClusteringFinisher
{
public:
    MPICoresetClusteringFinisher(AbstractClosestClusterUpdater* updater) : AbstractCoresetClusteringFinisher(updater){};

    virtual ~MPICoresetClusteringFinisher(){};

    value_t finishClustering(KmeansData* const kmeansData) override
    {
        pUpdater->findAndUpdateClosestClusters(kmeansData);

        MPI_Allgatherv(MPI_IN_PLACE, kmeansData->mLengths.at(kmeansData->mRank), MPI_INT,
                       kmeansData->pClustering->data(), kmeansData->mLengths.data(), kmeansData->mDisplacements.data(),
                       MPI_INT, MPI_COMM_WORLD);

        MPI_Allgatherv(MPI_IN_PLACE, kmeansData->mLengths.at(kmeansData->mRank), MPI_FLOAT,
                       kmeansData->pSqDistances->data(), kmeansData->mLengths.data(), kmeansData->mDisplacements.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        return std::accumulate(kmeansData->pSqDistances->begin(), kmeansData->pSqDistances->end(), 0.0);
    }
};