#include "ClosestClusterFinder.hpp"

ClosestCluster ClosestClusterFinder::findClosestCluster(value_t* datapoint,
                                                        std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    int clusterIdx, numExistingClusters = (*ppClusters)->getNumData();
    value_t minDistance = -1;

    for (int i = 0; i < numExistingClusters; i++)
    {
        value_t tempDistance = (*distanceFunc)(datapoint, (*ppClusters)->at(i), (*ppClusters)->getNumFeatures());
        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    return ClosestCluster{ clusterIdx, std::pow(minDistance, 2) };
}

ClosestCluster ClosestNewClusterFinder::findClosestCluster(value_t* datapoint,
                                                           std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    static int prevNumClusters, intermediate;
    int clusterIdx, numExistingClusters = (*ppClusters)->getNumData();
    value_t minDistance = -1;

    if (numExistingClusters == 1)
    {
        prevNumClusters = 0;
        intermediate    = 1;
    }

    for (int i = prevNumClusters; i < numExistingClusters; i++)
    {
        value_t tempDistance = (*distanceFunc)(datapoint, (*ppClusters)->at(i), (*ppClusters)->getNumFeatures());

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    if (intermediate != numExistingClusters)
    {
        prevNumClusters = intermediate;
        intermediate    = numExistingClusters;
    }

    return ClosestCluster{ clusterIdx, std::pow(minDistance, 2) };
}
