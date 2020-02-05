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
public:
    ~ClosestNewClusterFinder(){};
    ClosestCluster findClosestCluster(const int& dataIdx, KmeansData* const kmeansData) override;
};