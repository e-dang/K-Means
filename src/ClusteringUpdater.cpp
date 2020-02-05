#include "ClusteringUpdater.hpp"

void ClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster, KmeansData* const kmeansData)
{
    int displacedDataIdx   = kmeansData->mDisplacements.at(kmeansData->mRank) + dataIdx;
    int& clusterAssignment = kmeansData->pClustering->at(displacedDataIdx);
    value_t weight         = kmeansData->pWeights->at(dataIdx);

    if (kmeansData->pSqDistances->at(displacedDataIdx) > closestCluster.distance ||
        kmeansData->pSqDistances->at(displacedDataIdx) < 0)
    {
        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0 && kmeansData->pClusterWeights->at(clusterAssignment) > 0)
                kmeansData->pClusterWeights->at(clusterAssignment) -= weight;
            kmeansData->pClusterWeights->at(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->pSqDistances->at(displacedDataIdx) = closestCluster.distance;
    }
}

void AtomicClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster,
                                     KmeansData* const kmeansData)
{
    int displacedDataIdx   = kmeansData->mDisplacements.at(kmeansData->mRank) + dataIdx;
    int& clusterAssignment = kmeansData->pClustering->at(displacedDataIdx);
    value_t weight         = kmeansData->pWeights->at(dataIdx);

    if (kmeansData->pSqDistances->at(displacedDataIdx) > closestCluster.distance ||
        kmeansData->pSqDistances->at(displacedDataIdx) < 0)
    {
        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0 && kmeansData->pClusterWeights->at(clusterAssignment) > 0)
#pragma omp atomic
                kmeansData->pClusterWeights->at(clusterAssignment) -= weight;
#pragma omp atomic
            kmeansData->pClusterWeights->at(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->pSqDistances->at(displacedDataIdx) = closestCluster.distance;
    }
}

void DistributedClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster,
                                          KmeansData* const kmeansData)
{
    int displacedDataIdx   = kmeansData->mDisplacements.at(kmeansData->mRank) + dataIdx;
    int& clusterAssignment = kmeansData->pClustering->at(displacedDataIdx);
    value_t weight         = kmeansData->pWeights->at(dataIdx);

    if (kmeansData->pSqDistances->at(displacedDataIdx) > closestCluster.distance ||
        kmeansData->pSqDistances->at(displacedDataIdx) < 0)
    {
        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0) kmeansData->pClusterWeights->at(clusterAssignment) -= weight;
            kmeansData->pClusterWeights->at(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->pSqDistances->at(displacedDataIdx) = closestCluster.distance;
    }
}

void AtomicDistributedClusteringUpdater::update(const int& dataIdx, const ClosestCluster& closestCluster,
                                                KmeansData* const kmeansData)
{
    int displacedDataIdx   = kmeansData->mDisplacements.at(kmeansData->mRank) + dataIdx;
    int& clusterAssignment = kmeansData->pClustering->at(displacedDataIdx);
    value_t weight         = kmeansData->pWeights->at(dataIdx);

    if (kmeansData->pSqDistances->at(displacedDataIdx) > closestCluster.distance ||
        kmeansData->pSqDistances->at(displacedDataIdx) < 0)
    {
        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0)
#pragma omp atomic
                kmeansData->pClusterWeights->at(clusterAssignment) -= weight;
#pragma omp atomic
            kmeansData->pClusterWeights->at(closestCluster.clusterIdx) += weight;
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->pSqDistances->at(displacedDataIdx) = closestCluster.distance;
    }
}