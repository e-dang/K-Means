#pragma once

#include <memory>

#include "ClosestClusterFinder.hpp"
#include "ClusteringUpdater.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"

class AbstractClosestClusterUpdater
{
protected:
    std::unique_ptr<IClosestClusterFinder> pFinder;
    std::unique_ptr<AbstractClusteringDataUpdater> pUpdater;

public:
    AbstractClosestClusterUpdater(IClosestClusterFinder* finder, AbstractClusteringDataUpdater* updater) :
        pFinder(finder), pUpdater(updater)
    {
    }

    virtual ~AbstractClosestClusterUpdater() {}

    void findAndUpdateClosestCluster(const int_fast32_t& dataIdx, KmeansData* const kmeansData)
    {
        auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansData);
        pUpdater->update(dataIdx, closestCluster, kmeansData);
    }

    virtual void findAndUpdateClosestClusters(KmeansData* const kmeansData) = 0;
};

class SerialClosestClusterUpdater : public AbstractClosestClusterUpdater
{
public:
    SerialClosestClusterUpdater(IClosestClusterFinder* finder, AbstractClusteringDataUpdater* updater) :
        AbstractClosestClusterUpdater(finder, updater)
    {
    }

    ~SerialClosestClusterUpdater() {}

    void findAndUpdateClosestClusters(KmeansData* const kmeansData)
    {
        for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            findAndUpdateClosestCluster(i, kmeansData);
        }
    }
};

class OMPClosestClusterUpdater : public AbstractClosestClusterUpdater
{
public:
    OMPClosestClusterUpdater(IClosestClusterFinder* finder, AbstractClusteringDataUpdater* updater) :
        AbstractClosestClusterUpdater(finder, updater)
    {
    }

    ~OMPClosestClusterUpdater() {}

    void findAndUpdateClosestClusters(KmeansData* const kmeansData)
    {
#pragma omp parallel for shared(kmeansData), schedule(static)
        for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            findAndUpdateClosestCluster(i, kmeansData);
        }
    }
};