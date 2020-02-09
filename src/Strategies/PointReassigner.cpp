#include "Strategies/PointReassigner.hpp"

int_fast32_t AbstractPointReassigner::reassignPoint(const int_fast32_t& dataIdx, KmeansData* const kmeansData)
{
    auto before = kmeansData->clusteringAt(dataIdx);

    pUpdater->findAndUpdateClosestCluster(dataIdx, kmeansData);

    if (before != kmeansData->clusteringAt(dataIdx))
    {
        return 1;
    }
    return 0;
}

int_fast32_t SerialPointReassigner::reassignPoints(KmeansData* const kmeansData)
{
    int_fast32_t changed = 0;
    for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        changed += reassignPoint(i, kmeansData);
    }

    return changed;
}

int_fast32_t SerialOptimizedPointReassigner::reassignPoints(KmeansData* const kmeansData)
{
    int_fast32_t changed = 0;
    auto numFeatures     = kmeansData->pData->getNumFeatures();

    for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        auto clusterIdx = kmeansData->clusteringAt(i);
        auto dist       = std::pow(
          (*kmeansData->pDistanceFunc)(kmeansData->pData->at(i), kmeansData->pClusters->at(clusterIdx), numFeatures),
          2);
        if (dist > kmeansData->sqDistancesAt(i) || kmeansData->sqDistancesAt(i) < 0)
        {
            changed += reassignPoint(i, kmeansData);
        }
        else
        {
            kmeansData->sqDistancesAt(i) = dist;
        }
    }

    return changed;
}

int_fast32_t OMPPointReassigner::reassignPoints(KmeansData* const kmeansData)
{
    int_fast32_t changed = 0;

#pragma omp parallel for shared(kmeansData), schedule(static), reduction(+ : changed)
    for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        changed += reassignPoint(i, kmeansData);
    }

    return changed;
}

int_fast32_t OMPOptimizedPointReassigner::reassignPoints(KmeansData* const kmeansData)
{
    int_fast32_t changed = 0;
    auto numFeatures     = kmeansData->pData->getNumFeatures();

#pragma omp parallel for shared(kmeansData, numFeatures), schedule(static), reduction(+ : changed)
    for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
    {
        auto clusterIdx = kmeansData->clusteringAt(i);
        auto dist       = std::pow(
          (*kmeansData->pDistanceFunc)(kmeansData->pData->at(i), kmeansData->pClusters->at(clusterIdx), numFeatures),
          2);
        if (dist > kmeansData->sqDistancesAt(i) || kmeansData->sqDistancesAt(i) < 0)
        {
            changed += reassignPoint(i, kmeansData);
        }
        else
        {
            kmeansData->sqDistancesAt(i) = dist;
        }
    }

    return changed;
}