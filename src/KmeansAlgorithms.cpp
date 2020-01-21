#include "AbstractKmeansAlgorithms.hpp"
#include "mpi.h"

inline int AbstractKmeansAlgorithm::getCurrentNumClusters()
{
    return pClusters->data.size() / pClusters->numCols;
}

void AbstractKmeansAlgorithm::setUp(BundledAlgorithmData *bundledData)
{
    this->pMatrix = bundledData->matrix;
    this->pWeights = bundledData->weights;
}

void AbstractKmeansAlgorithm::setClusterData(ClusterData *clusterData)
{
    pClustering = &clusterData->clustering;
    pClusters = &clusterData->clusters;
    pClusterWeights = &clusterData->clusterWeights;
}

inline value_t AbstractKmeansAlgorithm::calcDistance(const int &dataIdx, const int &clusterIdx,
                                                     IDistanceFunctor *distanceFunc)
{
    return (*distanceFunc)(&*pMatrix->at(dataIdx), &*pClusters->at(clusterIdx), pClusters->numCols);
}

ClosestCluster AbstractKmeansAlgorithm::findClosestCluster(const int &dataIdx, IDistanceFunctor *distanceFunc)
{
    int clusterIdx, numExistingClusters = getCurrentNumClusters();
    value_t tempDistance, minDistance = -1;

    for (int i = 0; i < numExistingClusters; i++)
    {
        tempDistance = calcDistance(dataIdx, i, distanceFunc);

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx = i;
        }
    }

    return ClosestCluster{clusterIdx, minDistance};
}

inline void AbstractKmeansAlgorithm::updateClustering(const int &dataIdx, const int &clusterIdx)
{
    int &clusterAssignment = pClustering->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0 && pClusterWeights->at(clusterAssignment) > 0)
            pClusterWeights->at(clusterAssignment) -= pWeights->at(dataIdx);
        pClusterWeights->at(clusterIdx) += pWeights->at(dataIdx);
        clusterAssignment = clusterIdx;
    }
}

inline void AbstractOMPKmeansAlgorithm::updateClustering(const int &dataIdx, const int &clusterIdx)
{
    int &clusterAssignment = pClustering->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0 && pClusterWeights->at(clusterAssignment) > 0)
#pragma omp atomic
            pClusterWeights->at(clusterAssignment) -= pWeights->at(dataIdx);
#pragma omp atomic
        pClusterWeights->at(clusterIdx) += pWeights->at(dataIdx);
        clusterAssignment = clusterIdx;
    }
}

inline void AbstractMPIKmeansAlgorithm::updateClustering(const int &dataIdx, const int &clusterIdx)
{
    int &clusterAssignment = pClustering->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0)
            pClusterWeights->at(clusterAssignment) -= pWeights->at(dataIdx);
        pClusterWeights->at(clusterIdx) += pWeights->at(dataIdx);
        clusterAssignment = clusterIdx;
    }
}

void AbstractMPIKmeansAlgorithm::setUp(BundledAlgorithmData *bundledData)
{
    AbstractKmeansAlgorithm::setUp(bundledData);

    auto dataChunks = dynamic_cast<BundledMPIAlgorithmData *>(bundledData)->dataChunks;
    this->mRank = dataChunks->rank;
    this->pMatrixChunk = &dataChunks->matrixChunk;
    this->pLengths = &dataChunks->lengths;
    this->pDisplacements = &dataChunks->displacements;
}

void AbstractMPIKmeansAlgorithm::bcastClusterData()
{
    MPI_Bcast(pClustering->data(), pClustering->size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(pClusters->data.data(), pClusters->data.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

inline value_t AbstractMPIKmeansAlgorithm::calcDistance(const int &dataIdx, const int &clusterIdx,
                                                        IDistanceFunctor *distanceFunc)
{
    return (*distanceFunc)(&*pMatrixChunk->at(dataIdx), &*pClusters->at(clusterIdx), pClusters->numCols);
}