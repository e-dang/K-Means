#pragma once

#include <memory>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterFinder.hpp"
#include "Strategies/ClusteringUpdater.hpp"

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

    void findAndUpdateClosestCluster(const int32_t& dataIdx, KmeansData* const kmeansData);

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

    void findAndUpdateClosestClusters(KmeansData* const kmeansData) override;
};

class OMPClosestClusterUpdater : public AbstractClosestClusterUpdater
{
public:
    OMPClosestClusterUpdater(IClosestClusterFinder* finder, AbstractClusteringDataUpdater* updater) :
        AbstractClosestClusterUpdater(finder, updater)
    {
    }

    ~OMPClosestClusterUpdater() {}

    void findAndUpdateClosestClusters(KmeansData* const kmeansData) override;
};