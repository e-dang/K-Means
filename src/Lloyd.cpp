
#include "Lloyd.hpp"
#include "mpi.h"

void Lloyd::maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed;
    do
    {
        updateClusters();
        changed = reassignPoints(distances, distanceFunc);

    } while (changed > (pMatrix->numRows * MIN_PERCENT_CHANGED)); // do until 99.9% of data doesnt change
}

void Lloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(pClusters->data.begin(), pClusters->data.end(), 0);

    // calc the weighted sum of each feature for all points belonging to a cluster
    for (int i = 0; i < pMatrix->numRows; i++)
    {
        for (int j = 0; j < pMatrix->numCols; j++)
        {
            pClusters->at(pClustering->at(i), j) += pWeights->at(i) * pMatrix->at(i, j);
        }
    }

    // average out the weighted sum of each cluster based on the number of datapoints assigned to it
    for (int i = 0; i < pClusters->numRows; i++)
    {
        for (int j = 0; j < pClusters->numCols; j++)
        {
            pClusters->data.at(i * pClusters->numCols + j) /= pClusterWeights->at(i);
        }
    }
}

int Lloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < pMatrix->numRows; i++)
    {

        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != pClustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < pMatrix->numRows; i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        value_t dist = std::pow((*distanceFunc)(&*pMatrix->at(i), &*pClusters->at(pClustering->at(i)), pMatrix->numCols), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            // find closest cluster for each datapoint and update cluster assignment
            int before = pClustering->at(i);
            auto closestCluster = findClosestCluster(i, distanceFunc);
            updateClustering(i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);

            // check if cluster assignments have changed
            if (before != pClustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            distances->at(i) = dist;
        }
    }

    return changed;
}

void OMPLloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(pClusters->data.begin(), pClusters->data.end(), 0);

    // calc the weighted sum of each feature for all points belonging to a cluster
    for (int i = 0; i < pMatrix->numRows; i++)
    {
        for (int j = 0; j < pMatrix->numCols; j++)
        {
            pClusters->at(pClustering->at(i), j) += pWeights->at(i) * pMatrix->at(i, j);
        }
    }

    // average out the weighted sum of each cluster based on the number of datapoints assigned to it
#pragma omp parallel for schedule(static), collapse(2)
    for (int i = 0; i < pClusters->numRows; i++)
    {
        for (int j = 0; j < pClusters->numCols; j++)
        {
            pClusters->data.at(i * pClusters->numCols + j) /= pClusterWeights->at(i);
        }
    }
}

int OMPLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

#pragma omp parallel for shared(distances, distanceFunc), schedule(static), reduction(+ \
                                                                                      : changed)
    for (int i = 0; i < pMatrix->numRows; i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != pClustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OMPOptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

#pragma omp parallel for shared(distances, distanceFunc), schedule(static), reduction(+ \
                                                                                      : changed)
    for (int i = 0; i < pMatrix->numRows; i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        value_t dist = std::pow((*distanceFunc)(&*pMatrix->at(i), &*pClusters->at(pClustering->at(i)), pMatrix->numCols), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            // find closest cluster for each datapoint and update cluster assignment
            int before = pClustering->at(i);
            auto closestCluster = findClosestCluster(i, distanceFunc);
            updateClustering(i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);

            // check if cluster assignments have changed
            if (before != pClustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            distances->at(i) = dist;
        }
    }

    return changed;
}

void MPILloyd::updateClusters()
{
    std::vector<value_t> copyWeights(pClusterWeights->size());
    MPI_Reduce(pClusterWeights->data(), copyWeights.data(), copyWeights.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
        // reinitialize clusters
        std::fill(pClusters->data.begin(), pClusters->data.end(), 0);

        // calc the weighted sum of each feature for all points belonging to a cluster
        for (int i = 0; i < pMatrix->numRows; i++)
        {
            for (int j = 0; j < pMatrix->numCols; j++)
            {
                pClusters->at(pClustering->at(i), j) += pWeights->at(i) * pMatrix->at(i, j);
            }
        }

        // average out the weighted sum of each cluster based on the number of datapoints assigned to it
        for (int i = 0; i < pClusters->numRows; i++)
        {
            for (int j = 0; j < pClusters->numCols; j++)
            {
                pClusters->data.at(i * pClusters->numCols + j) /= copyWeights.at(i);
            }
        }
    }

    MPI_Bcast(pClusters->data.data(), pClusters->data.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int MPILloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

    for (int i = 0; i < pMatrixChunk->numRows; i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        AbstractMPIKmeansAlgorithm::updateClustering(pDisplacements->at(mRank) + i, closestCluster.clusterIdx);
        distances->at(pDisplacements->at(mRank) + i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != pClustering->at(i))
        {
            changed++;
        }
    }

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, distances->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);

    return changed;
}