
#include "Lloyd.hpp"
#include "mpi.h"
#include <iostream>

void TemplateLloyd::maximize()
{
    int changed;

    do
    {
        pClusters->fill(0);

        // calc the weighted sum of each feature for all points belonging to a cluster
        calcClusterSums();

        averageClusterSums();

        changed = reassignPoints();

    } while (changed > (mTotalNumData * MIN_PERCENT_CHANGED)); // do until 99.9% of data doesnt change
}

void TemplateLloyd::calcClusterSums()
{
    pAverager->calculateSum(pData, pClusters, pClustering, pWeights);
}

void TemplateLloyd::averageClusterSums()
{
    pAverager->normalizeSum(pClusters, pClusterWeights);
}

int TemplateLloyd::reassignPoints()
{
    int changed = 0;
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {

        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);
        findAndUpdateClosestCluster(i);

        // check if cluster assignments have changed
        if (before != pClustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OptimizedLloyd::reassignPoints()
{
    int changed = 0;
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        int clusterIdx = pClustering->at(pDisplacements->at(mRank) + i);
        value_t dist = std::pow((*pDistanceFunc)(pData->at(i), pClusters->at(clusterIdx), pData->getNumFeatures()), 2);
        if (dist > pDistances->at(i) || pDistances->at(i) < 0)
        {
            int before = pClustering->at(i);

            // find closest cluster for each datapoint and update cluster assignment
            findAndUpdateClosestCluster(i);

            // check if cluster assignments have changed
            if (before != pClustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            pDistances->at(i) = dist;
        }
    }

    return changed;
}

int OMPLloyd::reassignPoints()
{
    int changed = 0;

#pragma omp parallel for schedule(static), reduction(+ \
                                                     : changed)
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        int before = pClustering->at(i);

        findAndUpdateClosestCluster(i);

        // check if cluster assignments have changed
        if (before != pClustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OMPOptimizedLloyd::reassignPoints()
{
    int changed = 0;

#pragma omp parallel for schedule(static), reduction(+ \
                                                     : changed)
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        int clusterIdx = pClustering->at(pDisplacements->at(mRank) + i);
        value_t dist = std::pow((*pDistanceFunc)(pData->at(i), pClusters->at(clusterIdx), pData->getNumFeatures()), 2);
        if (dist > pDistances->at(i) || pDistances->at(i) < 0)
        {
            int before = pClustering->at(i);

            // find closest cluster for each datapoint and update cluster assignment
            findAndUpdateClosestCluster(i);

            // check if cluster assignments have changed
            if (before != pClustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            pDistances->at(i) = dist;
        }
    }

    return changed;
}

void MPILloyd::calcClusterSums()
{
    // TemplateLloyd::calcClusterSums();
    for (int i = 0; i < pData->getNumData(); i++)
    {
        for (int j = 0; j < pData->getNumFeatures(); j++)
        {
            pClusters->at(pClustering->at(pDisplacements->at(mRank) + i), j) += pWeights->at(i) * pData->at(i, j);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, pClusters->data(), pClusters->size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void MPILloyd::averageClusterSums()
{
    std::vector<value_t> copyWeights(pClusterWeights->size());
    MPI_Reduce(pClusterWeights->data(), copyWeights.data(), copyWeights.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
        pAverager->normalizeSum(pClusters, &copyWeights);
    }

    MPI_Bcast(pClusters->data(), pClusters->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int MPILloyd::reassignPoints()
{
    int changed = 0;
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = pClustering->at(i);

        findAndUpdateClosestCluster(i);

        // check if cluster assignments have changed
        if (before != pClustering->at(i))
        {
            changed++;
        }
    }

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pDistances->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);

    return changed;
}

int MPIOptimizedLloyd::reassignPoints()
{
    int changed = 0;
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        int clusterIdx = pClustering->at(pDisplacements->at(mRank) + i);
        value_t dist = std::pow((*pDistanceFunc)(pData->at(i), pClusters->at(clusterIdx), pData->getNumFeatures()), 2);
        if (dist > pDistances->at(i) || pDistances->at(i) < 0)
        {
            int before = pClustering->at(i);

            // find closest cluster for each datapoint and update cluster assignment
            findAndUpdateClosestCluster(i);

            // check if cluster assignments have changed
            if (before != pClustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            pDistances->at(i) = dist;
        }
    }

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pDistances->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);

    return changed;
}