#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IClosestClusterFinder
{
public:
    virtual ~IClosestClusterFinder(){};

    virtual ClosestCluster findClosestCluster(const int_fast32_t& dataIdx, KmeansData* const kmeansData) = 0;
};

class ClosestClusterFinder : public IClosestClusterFinder
{
public:
    ~ClosestClusterFinder(){};

    ClosestCluster findClosestCluster(const int_fast32_t& dataIdx, KmeansData* const kmeansData) override;
};

class ClosestNewClusterFinder : public IClosestClusterFinder
{
public:
    ~ClosestNewClusterFinder(){};

    ClosestCluster findClosestCluster(const int_fast32_t& dataIdx, KmeansData* const kmeansData) override;
};