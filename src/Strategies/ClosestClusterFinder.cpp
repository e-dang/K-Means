#include "Strategies/ClosestClusterFinder.hpp"

ClosestCluster ClosestClusterFinder::findClosestCluster(const int32_t& dataIdx, KmeansData* const kmeansData)
{
    auto numFeatures         = kmeansData->pData->getNumFeatures();
    auto numExistingClusters = kmeansData->pClusters->getNumData();
    const auto datapoint     = kmeansData->pData->at(dataIdx);
    double minDistance       = -1.0;
    int32_t clusterIdx       = -1;

    for (int32_t i = 0; i < numExistingClusters; i++)
    {
        double tempDistance = (*kmeansData->pDistanceFunc)(datapoint, kmeansData->pClusters->at(i), numFeatures);

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    return ClosestCluster{ clusterIdx, std::pow(minDistance, 2) };
}

ClosestCluster ClosestNewClusterFinder::findClosestCluster(const int32_t& dataIdx, KmeansData* const kmeansData)
{
    thread_local static int32_t prevNumClusters;
    thread_local static int32_t intermediate;
    auto numFeatures         = kmeansData->pData->getNumFeatures();
    auto numExistingClusters = kmeansData->pClusters->getNumData();
    const auto datapoint     = kmeansData->pData->at(dataIdx);
    double minDistance       = -1.0;
    int32_t clusterIdx       = -1;

    if (numExistingClusters == 1)
    {
        prevNumClusters = 0;
        intermediate    = 1;
    }

    for (auto i = prevNumClusters; i < numExistingClusters; i++)
    {
        double tempDistance = (*kmeansData->pDistanceFunc)(datapoint, kmeansData->pClusters->at(i), numFeatures);

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
