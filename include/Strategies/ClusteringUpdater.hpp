#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class AbstractClusteringDataUpdater
{
public:
    virtual ~AbstractClusteringDataUpdater(){};

    virtual void update(const int32_t& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData);

    virtual void updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment,
                                      const int32_t& newAssignment, KmeansData* const kmeansData) = 0;
};

class ClusteringDataUpdater : public AbstractClusteringDataUpdater
{
public:
    ~ClusteringDataUpdater() {}

    void updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment, const int32_t& newAssignment,
                              KmeansData* const kmeansData) override;
};

class AtomicClusteringDataUpdater : public AbstractClusteringDataUpdater
{
public:
    ~AtomicClusteringDataUpdater() {}

    void updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment, const int32_t& newAssignment,
                              KmeansData* const kmeansData) override;
};

class DistributedClusteringDataUpdater : public AbstractClusteringDataUpdater
{
public:
    ~DistributedClusteringDataUpdater() {}

    void updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment, const int32_t& newAssignment,
                              KmeansData* const kmeansData) override;
};

class AtomicDistributedClusteringDataUpdater : public AbstractClusteringDataUpdater
{
public:
    ~AtomicDistributedClusteringDataUpdater() {}

    void updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment, const int32_t& newAssignment,
                              KmeansData* const kmeansData);
};

class CoresetClusteringDataUpdater : public AbstractClusteringDataUpdater
{
public:
    ~CoresetClusteringDataUpdater() {}

    void updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment, const int32_t& newAssignment,
                              KmeansData* const kmeansData) override;
};