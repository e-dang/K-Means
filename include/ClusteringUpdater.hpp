#pragma once

#include <vector>

#include "Definitions.hpp"

class AbstractClusteringUpdater
{
protected:
    std::vector<int>** ppClustering;
    std::vector<value_t>** ppClusterWeights;

public:
    AbstractClusteringUpdater(std::vector<int>** clustering, std::vector<value_t>** clusterWeights) :
        ppClustering(clustering), ppClusterWeights(clusterWeights)
    {
    }

    virtual void update(const int& dataIdx, const int& clusterIdx, const value_t& weight) = 0;
};

class ClusteringUpdater : public AbstractClusteringUpdater
{
public:
    ClusteringUpdater(std::vector<int>** clustering, std::vector<value_t>** clusterWeights) :
        AbstractClusteringUpdater(clustering, clusterWeights)
    {
    }

    void update(const int& dataIdx, const int& clusterIdx, const value_t& weight) override;
};

class AtomicClusteringUpdater : public AbstractClusteringUpdater
{
public:
    AtomicClusteringUpdater(std::vector<int>** clustering, std::vector<value_t>** clusterWeights) :
        AbstractClusteringUpdater(clustering, clusterWeights)
    {
    }

    void update(const int& dataIdx, const int& clusterIdx, const value_t& weight) override;
};

class DistributedClusteringUpdater : public AbstractClusteringUpdater
{
public:
    DistributedClusteringUpdater(std::vector<int>** clustering, std::vector<value_t>** clusterWeights) :
        AbstractClusteringUpdater(clustering, clusterWeights)
    {
    }

    void update(const int& dataIdx, const int& clusterIdx, const value_t& weight) override;
};