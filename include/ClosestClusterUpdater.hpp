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
    std::unique_ptr<IClusteringUpdater> pUpdater;

public:
    AbstractClosestClusterUpdater(IClosestClusterFinder* finder, IClusteringUpdater* updater) :
        pFinder(finder), pUpdater(updater)
    {
    }

    virtual ~AbstractClosestClusterUpdater() {}

    void findAndUpdateClosestCluster(const int& dataIdx, KmeansData* const kmeansData)
    {
        auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansData);
        pUpdater->update(dataIdx, closestCluster, kmeansData);
    }

    virtual void findAndUpdateClosestClusters(KmeansData* const kmeansData) = 0;
};

class SerialClosestClusterUpdater : public AbstractClosestClusterUpdater
{
public:
    SerialClosestClusterUpdater(IClosestClusterFinder* finder, IClusteringUpdater* updater) :
        AbstractClosestClusterUpdater(finder, updater)
    {
    }

    ~SerialClosestClusterUpdater() {}

    void findAndUpdateClosestClusters(KmeansData* const kmeansData)
    {
        for (int i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            findAndUpdateClosestCluster(i, kmeansData);
        }
    }
};

class OMPClosestClusterUpdater : public AbstractClosestClusterUpdater
{
public:
    OMPClosestClusterUpdater(IClosestClusterFinder* finder, IClusteringUpdater* updater) :
        AbstractClosestClusterUpdater(finder, updater)
    {
    }

    ~OMPClosestClusterUpdater() {}

    void findAndUpdateClosestClusters(KmeansData* const kmeansData)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            findAndUpdateClosestCluster(i, kmeansData);
        }
    }
};