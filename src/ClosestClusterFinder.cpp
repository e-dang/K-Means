#include "ClosestClusterFinder.hpp"

ClosestCluster ClosestClusterFinder::findClosestCluster(const int_fast32_t& dataIdx, KmeansData* const kmeansData)
{
    int_fast32_t clusterIdx;
    auto numFeatures         = kmeansData->pData->getNumFeatures();
    auto numExistingClusters = kmeansData->pClusters->getNumData();
    const auto datapoint     = kmeansData->pData->at(dataIdx);
    double minDistance       = -1.0;

    for (int_fast32_t i = 0; i < numExistingClusters; i++)
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

ClosestCluster ClosestNewClusterFinder::findClosestCluster(const int_fast32_t& dataIdx, KmeansData* const kmeansData)
{
    thread_local static int_fast32_t prevNumClusters;
    thread_local static int_fast32_t intermediate;
    int_fast32_t clusterIdx;
    auto numFeatures         = kmeansData->pData->getNumFeatures();
    auto numExistingClusters = kmeansData->pClusters->getNumData();
    const auto datapoint     = kmeansData->pData->at(dataIdx);
    double minDistance       = -1.0;

    if (numExistingClusters == 1)
    {
        prevNumClusters = 0;
        intermediate    = 1;
    }

    for (auto i = prevNumClusters; i < numExistingClusters; i++)
    {
        auto tempDistance = (*kmeansData->pDistanceFunc)(datapoint, kmeansData->pClusters->at(i), numFeatures);

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
