#include "ClosestClusterFinder.hpp"

ClosestCluster ClosestClusterFinder::findClosestCluster(const int& dataIdx, KmeansData* const kmeansData)
{
    int clusterIdx;
    int numFeatures          = kmeansData->pData->getNumFeatures();
    int numExistingClusters  = kmeansData->pClusters->getNumData();
    const value_t* datapoint = kmeansData->pData->at(dataIdx);
    value_t minDistance      = -1;

    for (int i = 0; i < numExistingClusters; i++)
    {
        value_t tempDistance = (*kmeansData->pDistanceFunc)(datapoint, kmeansData->pClusters->at(i), numFeatures);

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    return ClosestCluster{ clusterIdx, std::pow(minDistance, 2) };
}

ClosestCluster ClosestNewClusterFinder::findClosestCluster(const int& dataIdx, KmeansData* const kmeansData)
{
    int clusterIdx;
    int numFeatures          = kmeansData->pData->getNumFeatures();
    int numExistingClusters  = kmeansData->pClusters->getNumData();
    const value_t* datapoint = kmeansData->pData->at(dataIdx);
    value_t minDistance      = -1;

    if (numExistingClusters == 1)
    {
        resetState(numExistingClusters);
    }

    for (int i = prevNumClusters; i < numExistingClusters; i++)
    {
        value_t tempDistance = (*kmeansData->pDistanceFunc)(datapoint, kmeansData->pClusters->at(i), numFeatures);

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    if (intermediate != numExistingClusters)
    {
        updateState(numExistingClusters);
    }

    return ClosestCluster{ clusterIdx, std::pow(minDistance, 2) };
}
