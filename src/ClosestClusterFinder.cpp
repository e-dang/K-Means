#include "ClosestClusterFinder.hpp"

ClosestCluster ClosestClusterFinder::findClosestCluster(value_t *datapoint, IDistanceFunctor *distanceFunc)
{
    int clusterIdx, numExistingClusters = (*ppClusters)->getNumData();
    value_t tempDistance, minDistance = -1;

    for (int i = 0; i < numExistingClusters; i++)
    {
        tempDistance = (*distanceFunc)(datapoint, (*ppClusters)->at(i), (*ppClusters)->getNumFeatures());

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx = i;
        }
    }

    return ClosestCluster{clusterIdx, minDistance};
}

ClosestCluster ClosestNewClusterFinder::findClosestCluster(value_t *datapoint, IDistanceFunctor *distanceFunc)
{
    static int prevNumClusters = 0;
    int clusterIdx, numExistingClusters = (*ppClusters)->getNumData();
    value_t tempDistance, minDistance = -1;

    for (int i = prevNumClusters; i < numExistingClusters; i++)
    {
        tempDistance = (*distanceFunc)(datapoint, (*ppClusters)->at(i), (*ppClusters)->getNumFeatures());

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx = i;
        }
    }

    prevNumClusters = numExistingClusters;

    return ClosestCluster{clusterIdx, minDistance};
}
