#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"

class IClosestClusterFinder
{
public:
    virtual ~IClosestClusterFinder(){};

    virtual ClosestCluster findClosestCluster(const int& dataIdx, KmeansData* const kmeansData) = 0;
};

class ClosestClusterFinder : public IClosestClusterFinder
{
public:
    ~ClosestClusterFinder(){};

    ClosestCluster findClosestCluster(const int& dataIdx, KmeansData* const kmeansData) override;
};

class ClosestNewClusterFinder : public IClosestClusterFinder
{
private:
    unsigned int prevNumClusters;
    unsigned int intermediate;

public:
    ~ClosestNewClusterFinder(){};

    ClosestCluster findClosestCluster(const int& dataIdx, KmeansData* const kmeansData) override;

    void resetState(const int& numExistingClusters)
    {
        if (numExistingClusters == 1)
        {
            prevNumClusters = 0;
            intermediate    = 1;
        }
    }

    void updateState(const int& numExistingClusters)
    {
        prevNumClusters = intermediate;
        intermediate    = numExistingClusters;
    }
};