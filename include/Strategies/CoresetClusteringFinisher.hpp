#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"

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

    value_t finishClustering(KmeansData* const kmeansData) override;
};

class MPICoresetClusteringFinisher : public AbstractCoresetClusteringFinisher
{
public:
    MPICoresetClusteringFinisher(AbstractClosestClusterUpdater* updater) : AbstractCoresetClusteringFinisher(updater){};

    virtual ~MPICoresetClusteringFinisher(){};

    value_t finishClustering(KmeansData* const kmeansData) override;
};