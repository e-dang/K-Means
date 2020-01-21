#include "AbstractKmeansAlgorithms.hpp"
#include "mpi.h"
#include "Utils.hpp"

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

void AbstractKmeansInitializer::appendCluster(const int &dataIdx)
{
    std::copy(pMatrix->at(dataIdx), pMatrix->at(dataIdx) + pMatrix->numCols, std::back_inserter(pClusters->data));
}

void AbstractKmeansMaximizer::addPointToCluster(const int &dataIdx)
{
    for (int j = 0; j < pMatrix->numCols; j++)
    {
        pClusters->at(pClustering->at(dataIdx), j) += pWeights->at(dataIdx) * pMatrix->at(dataIdx, j);
    }
}

void AbstractKmeansMaximizer::averageCluster(const int &clusterIdx)
{
    for (int j = 0; j < pClusters->numCols; j++)
    {
        pClusters->data.at(clusterIdx * pClusters->numCols + j) /= pClusterWeights->at(clusterIdx);
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
    return (*distanceFunc)(&*pMatrixChunk->at(dataIdx - pDisplacements->at(mRank)), &*pClusters->at(clusterIdx), pClusters->numCols);
}

void WeightedClusterSelection::weightedClusterSelection(std::vector<value_t> *distances, float &randSumFrac)
{
    int randIdx = weightedRandomSelection(distances, randSumFrac);
    pAlg->appendCluster(randIdx);
    pAlg->updateClustering(randIdx, pAlg->getCurrentNumClusters() - 1);
}

void FindAndUpdateClosestCluster::findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
                                                              IDistanceFunctor *distanceFunc)
{
    auto closestCluster = pAlg->findClosestCluster(dataIdx, distanceFunc);
    pAlg->updateClustering(dataIdx, closestCluster.clusterIdx);
    distances->at(dataIdx) = std::pow(closestCluster.distance, 2);
}

void OptFindAndUpdateClosestCluster::findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
                                                                 IDistanceFunctor *distanceFunc)
{
    int clusterIdx = pAlg->getCurrentNumClusters() - 1;
    value_t newDist = pAlg->calcDistance(dataIdx, clusterIdx, distanceFunc);
    if (newDist < distances->at(dataIdx) || distances->at(dataIdx) < 0)
    {
        pAlg->updateClustering(dataIdx, clusterIdx);
        distances->at(dataIdx) = std::pow(newDist, 2);
    }
}

void ReassignmentClosestClusterFinder::findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
                                                                   IDistanceFunctor *distanceFunc)
{
    // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
    value_t dist = std::pow(pAlg->calcDistance(dataIdx, pClustering->at(dataIdx), distanceFunc), 2);
    if (dist > distances->at(dataIdx) || distances->at(dataIdx) < 0)
    {
        // find closest cluster for each datapoint and update cluster assignment
        auto closestCluster = pAlg->findClosestCluster(dataIdx, distanceFunc);
        pAlg->updateClustering(dataIdx, closestCluster.clusterIdx);
        distances->at(dataIdx) = std::pow(closestCluster.distance, 2);
    }
}

// void CalculateClusterMeans::calculateSum()
// {
//     // reinitialize clusters
//     std::fill(clusters->data.begin(), clusters->data.end(), 0);

//     // calc the weighted sum of each feature for all points belonging to a cluster
//     int numData = pAlg->getNumData(), numFeatures = pAlg->getNumFeatures();
//     for (int i = 0; i < numData; i++)
//     {
//         int clusterIdx = getClusteringAt(i);
//         value_t weight = getWeightsAt(i);
//         for (int j = 0; j < numFeatures; j++)
//         {
//             clusters->at(clusterIdx, j) += weight * getDataValAt(i, j);
//         }
//     }
// }

// void CalculateClusterMeans::calculateMean()
// {

//     // average out the weighted sum of each cluster based on the number of datapoints assigned to it
//     for (int i = 0; i < clusters->numRows; i++)
//     {
//         for (int j = 0; j < numFeatures; j++)
//         {
//             clusters->data.at(i * numFeatures + j) /= clusterWeights->at(i);
//         }
//     }
// }