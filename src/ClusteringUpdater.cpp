#include "ClusteringUpdater.hpp"

void ClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData)
{
    if (kmeansData->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansData->sqDistancesAt(dataIdx) < 0)
    {
        int& clusterAssignment = kmeansData->clusteringAt(dataIdx);
        value_t weight         = kmeansData->pWeights->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0 && kmeansData->clusterWeightsAt(clusterAssignment) > 0)
                kmeansData->clusterWeightsAt(clusterAssignment) -= weight;
            kmeansData->clusterWeightsAt(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}

void AtomicClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster,
                                     KmeansData* const kmeansData)
{
    if (kmeansData->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansData->sqDistancesAt(dataIdx) < 0)
    {
        int& clusterAssignment = kmeansData->clusteringAt(dataIdx);
        value_t weight         = kmeansData->pWeights->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0 && kmeansData->clusterWeightsAt(clusterAssignment) > 0)
#pragma omp atomic
                kmeansData->clusterWeightsAt(clusterAssignment) -= weight;
#pragma omp atomic
            kmeansData->clusterWeightsAt(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}

void DistributedClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster,
                                          KmeansData* const kmeansData)
{
    if (kmeansData->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansData->sqDistancesAt(dataIdx) < 0)
    {
        int& clusterAssignment = kmeansData->clusteringAt(dataIdx);
        value_t weight         = kmeansData->pWeights->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0) kmeansData->clusterWeightsAt(clusterAssignment) -= weight;
            kmeansData->clusterWeightsAt(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}

void AtomicDistributedClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster,
                                                KmeansData* const kmeansData)
{
    if (kmeansData->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansData->sqDistancesAt(dataIdx) < 0)
    {
        int& clusterAssignment = kmeansData->clusteringAt(dataIdx);
        value_t weight         = kmeansData->pWeights->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0)
#pragma omp atomic
                kmeansData->clusterWeightsAt(clusterAssignment) -= weight;
#pragma omp atomic
            kmeansData->clusterWeightsAt(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}