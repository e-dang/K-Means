#pragma once

#include "Definitions.hpp"
#include "DataClasses.hpp"
#include "DistanceFunctors.hpp"

class AbstractClosestClusterFinder
{
protected:
    Matrix **ppClusters;

public:
    AbstractClosestClusterFinder(Matrix **clusters) : ppClusters(clusters) {}

    virtual ClosestCluster findClosestCluster(value_t *datapoint, IDistanceFunctor *distanceFunc) = 0;
};

class ClosestClusterFinder : public AbstractClosestClusterFinder
{
public:
    ClosestClusterFinder(Matrix **clusters) : AbstractClosestClusterFinder(clusters) {}

    ClosestCluster findClosestCluster(value_t *datapoint, IDistanceFunctor *distanceFunc) override;
};

class ClosestNewClusterFinder : public AbstractClosestClusterFinder
{
public:
    ClosestNewClusterFinder(Matrix **clusters) : AbstractClosestClusterFinder(clusters) {}

    ClosestCluster findClosestCluster(value_t *datapoint, IDistanceFunctor *distanceFunc) override;
};