
#include "Lloyd.hpp"
#include "mpi.h"
#include <iostream>

void TemplateLloyd::maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed, numData = pMatrix->numRows;

    do
    {
        // reinitialize clusters
        std::fill(pClusters->data.begin(), pClusters->data.end(), 0);

        // calc the weighted sum of each feature for all points belonging to a cluster
        calcClusterSums();

        averageClusterSums();

        changed = reassignPoints(distances, distanceFunc);

    } while (changed > (numData * MIN_PERCENT_CHANGED)); // do until 99.9% of data doesnt change
}

void TemplateLloyd::calcClusterSums()
{

    for (int i = 0; i < pMatrix->numRows; i++)
    {
        addPointToCluster(i);
    }
}

void TemplateLloyd::averageClusterSums()
{
    for (int i = 0; i < pClusters->numRows; i++)
    {
        averageCluster(i);
    }
}

int TemplateLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < pMatrix->numRows; i++)
    {

        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);

        pFinder->findAndUpdateClosestCluster(i, distances, distanceFunc);

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
        value_t dist = std::pow(calcDistance(i, pClustering->at(i), distanceFunc), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            int before = pClustering->at(i);

            // find closest cluster for each datapoint and update cluster assignment
            pFinder->findAndUpdateClosestCluster(i, distances, distanceFunc);

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

void OMPLloyd::averageClusterSums()
{
    int numClusters = pClusters->numRows;
#pragma omp parallel for shared(numClusters), schedule(static)
    for (int i = 0; i < numClusters; i++)
    {
        averageCluster(i);
    }
}

int OMPLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0, numData = pMatrix->numRows;

#pragma omp parallel for shared(numData, distances, distanceFunc), schedule(static), reduction(+ \
                                                                                               : changed)
    for (int i = 0; i < numData; i++)
    {
        int before = pClustering->at(i);

        // find closest cluster for each datapoint and update cluster assignment
        pFinder->findAndUpdateClosestCluster(i, distances, distanceFunc);

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
        value_t dist = std::pow(calcDistance(i, pClustering->at(i), distanceFunc), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            int before = pClustering->at(i);

            // find closest cluster for each datapoint and update cluster assignment
            pFinder->findAndUpdateClosestCluster(i, distances, distanceFunc);

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

void MPILloyd::calcClusterSums()
{
    int numData = pMatrixChunk->numRows;
    for (int i = 0; i < numData; i++)
    {
        int dataIdx = pDisplacements->at(mRank) + i;
        addPointToCluster(dataIdx);
    }

    MPI_Allreduce(MPI_IN_PLACE, pClusters->data.data(), pClusters->data.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void MPILloyd::averageClusterSums()
{
    std::vector<value_t> copyWeights(pClusterWeights->size());
    MPI_Reduce(pClusterWeights->data(), copyWeights.data(), copyWeights.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
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
    int changed = 0, numData = pMatrixChunk->numRows;

    for (int i = 0; i < numData; i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);
        pFinder->findAndUpdateClosestCluster(pDisplacements->at(mRank) + i, distances, distanceFunc);

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

int MPIOptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0, numData = pMatrixChunk->numRows;

    for (int i = 0; i < numData; i++)
    {

        value_t dist = std::pow(calcDistance(pDisplacements->at(mRank) + i, pClustering->at(i), distanceFunc), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            int before = pClustering->at(i);

            // find closest cluster for each datapoint and update cluster assignment
            pFinder->findAndUpdateClosestCluster(pDisplacements->at(mRank) + i, distances, distanceFunc);

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

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, distances->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);

    return changed;
}