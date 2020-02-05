#pragma once

#include <vector>

#include "DataClasses.hpp"
#include "Definitions.hpp"

class IClusteringUpdater
{
public:
    virtual ~IClusteringUpdater(){};

    virtual void update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData) = 0;
};

class ClusteringUpdater : public IClusteringUpdater
{
public:
    ~ClusteringUpdater() {}

    void update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData) override;
};

class AtomicClusteringUpdater : public IClusteringUpdater
{
public:
    ~AtomicClusteringUpdater() {}

    void update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData) override;
};

class DistributedClusteringUpdater : public IClusteringUpdater
{
public:
    ~DistributedClusteringUpdater() {}

    void update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData) override;
};

class AtomicDistributedClusteringUpdater : public IClusteringUpdater
{
public:
    ~AtomicDistributedClusteringUpdater() {}

    void update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData) override;
};