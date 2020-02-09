#include "Strategies/ClosestClusterUpdater.hpp"

void AbstractClosestClusterUpdater::findAndUpdateClosestCluster(const int32_t& dataIdx, KmeansData* const kmeansData)
{
    auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansData);
    pUpdater->update(dataIdx, closestCluster, kmeansData);
}

void SerialClosestClusterUpdater::findAndUpdateClosestClusters(KmeansData* const kmeansData)
{
    for (int32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        findAndUpdateClosestCluster(i, kmeansData);
    }
}

void OMPClosestClusterUpdater::findAndUpdateClosestClusters(KmeansData* const kmeansData)
{
#pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        findAndUpdateClosestCluster(i, kmeansData);
    }
}