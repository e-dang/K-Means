#include "Strategies/ClusteringUpdater.hpp"

void AbstractClusteringDataUpdater::update(const int32_t& dataIdx, const ClosestCluster& closestCluster,
                                           KmeansData* const kmeansData)
{
    if (kmeansData->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansData->sqDistancesAt(dataIdx) < 0)
    {
        int32_t& clusterAssignment = kmeansData->clusteringAt(dataIdx);
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            updateClusterWeights(dataIdx, clusterAssignment, closestCluster.clusterIdx, kmeansData);
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}

void ClusteringDataUpdater::updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment,
                                                 const int32_t& newAssignment, KmeansData* const kmeansData)
{
    value_t weight = kmeansData->pWeights->at(dataIdx);
    if (prevAssignment >= 0 && kmeansData->clusterWeightsAt(prevAssignment) > 0)
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

void AtomicClusteringDataUpdater::updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment,
                                                       const int32_t& newAssignment, KmeansData* const kmeansData)
{
    value_t weight = kmeansData->pWeights->at(dataIdx);
    if (prevAssignment >= 0 && kmeansData->clusterWeightsAt(prevAssignment) > 0)
#pragma omp atomic
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
#pragma omp atomic
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

void DistributedClusteringDataUpdater::updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment,
                                                            const int32_t& newAssignment, KmeansData* const kmeansData)
{
    value_t weight = kmeansData->pWeights->at(dataIdx);
    if (prevAssignment >= 0)
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

void AtomicDistributedClusteringDataUpdater::updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment,
                                                                  const int32_t& newAssignment,
                                                                  KmeansData* const kmeansData)
{
    value_t weight = kmeansData->pWeights->at(dataIdx);
    if (prevAssignment >= 0)
#pragma omp atomic
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
#pragma omp atomic
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

void CoresetClusteringDataUpdater::updateClusterWeights(const int32_t& dataIdx, const int32_t& prevAssignment,
                                                        const int32_t& newAssignment, KmeansData* const kmeansData)
{
    // no operations here
}