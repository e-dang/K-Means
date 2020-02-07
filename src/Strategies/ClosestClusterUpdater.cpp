#include "Strategies/ClosestClusterUpdater.hpp"

void AbstractClosestClusterUpdater::findAndUpdateClosestCluster(const int_fast32_t& dataIdx,
                                                                KmeansData* const kmeansData)
{
    auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansData);
    pUpdater->update(dataIdx, closestCluster, kmeansData);
}

void SerialClosestClusterUpdater::findAndUpdateClosestClusters(KmeansData* const kmeansData)
{
    for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        findAndUpdateClosestCluster(i, kmeansData);
    }
}

void OMPClosestClusterUpdater::findAndUpdateClosestClusters(KmeansData* const kmeansData)
{
#pragma omp parallel for shared(kmeansData), schedule(static)
    for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        findAndUpdateClosestCluster(i, kmeansData);
    }
}