#include "ClusteringUpdater.hpp"

void ClusteringUpdater::update(const int& dataIdx, const int& clusterIdx, const value_t& weight)
{
    int& clusterAssignment = (*ppClustering)->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0 && (*ppClusterWeights)->at(clusterAssignment) > 0)
            (*ppClusterWeights)->at(clusterAssignment) -= weight;
        (*ppClusterWeights)->at(clusterIdx) += weight;
        clusterAssignment = clusterIdx;
    }
}

void AtomicClusteringUpdater::update(const int& dataIdx, const int& clusterIdx, const value_t& weight)
{
    int& clusterAssignment = (*ppClustering)->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0 && (*ppClusterWeights)->at(clusterAssignment) > 0)
#pragma omp atomic
            (*ppClusterWeights)->at(clusterAssignment) -= weight;
#pragma omp atomic
        (*ppClusterWeights)->at(clusterIdx) += weight;
        clusterAssignment = clusterIdx;
    }
}

void DistributedClusteringUpdater::update(const int& dataIdx, const int& clusterIdx, const value_t& weight)
{
    int& clusterAssignment = (*ppClustering)->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0) (*ppClusterWeights)->at(clusterAssignment) -= weight;
        (*ppClusterWeights)->at(clusterIdx) += weight;
        clusterAssignment = clusterIdx;
    }
}

void AtomicDistributedClusteringUpdater::update(const int& dataIdx, const int& clusterIdx, const value_t& weight)
{
    int& clusterAssignment = (*ppClustering)->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0)
#pragma omp atomic
            (*ppClusterWeights)->at(clusterAssignment) -= weight;
#pragma omp atomic
        (*ppClusterWeights)->at(clusterIdx) += weight;
        clusterAssignment = clusterIdx;
    }
}