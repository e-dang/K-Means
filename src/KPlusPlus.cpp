#include "KPlusPlus.hpp"

#include <omp.h>

#include "Utils.hpp"
#include "mpi.h"

void TemplateKPlusPlus::initialize()
{
    // initialize first cluster uniformly at random. Thus distances should be filled with same number i.e. 1
    weightedClusterSelection();

    // change fill distances vector with -1 so values aren't confused with actual distances
    std::fill(pDistances->begin(), pDistances->end(), -1);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int i = 1; i < pClusters->getMaxNumData(); i++)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestClusters();

        // select point to be next cluster center weighted by nearest distance squared
        weightedClusterSelection();
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestClusters();
}

void TemplateKPlusPlus::weightedClusterSelection()
{
    value_t randSumFrac = getRandFloat01() * std::accumulate(pDistances->begin(), pDistances->end(), 0);
    int dataIdx         = pSelector->select(pDistances, randSumFrac);
    pClusters->appendDataPoint(pData->at(dataIdx));
}

void TemplateKPlusPlus::findAndUpdateClosestClusters()
{
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        findAndUpdateClosestCluster(i);
    }
}

void OMPKPlusPlus::findAndUpdateClosestClusters()
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        findAndUpdateClosestCluster(i);
    }
}

void MPIKPlusPlus::weightedClusterSelection()
{
    int dataIdx;
    if (mRank == 0)
    {
        value_t randSumFrac = getRandFloat01MPI() * std::accumulate(pDistances->begin(), pDistances->end(), 0);
        dataIdx             = pSelector->select(pDistances, randSumFrac);
    }

    MPI_Bcast(&dataIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // find which rank holds the selected datapoint
    int rank, lengthSum = 0;
    for (rank = 0; rank < pLengths->size(); rank++)
    {
        lengthSum += pLengths->at(rank);
        if (lengthSum > dataIdx)
        {
            break;
        }
    }

    if (mRank == rank)
    {
        pClusters->appendDataPoint(pData->at(dataIdx - pDisplacements->at(mRank)));
    }
    else
    {
        pClusters->reserve(pClusters->getNumData() + 1);
    }

    MPI_Bcast(pClustering->data(), pClustering->size(), MPI_INT, rank, MPI_COMM_WORLD);
    MPI_Bcast(pClusters->data(), pClusters->size(), MPI_FLOAT, rank, MPI_COMM_WORLD);
}

void MPIKPlusPlus::findAndUpdateClosestClusters()
{
    for (int i = 0; i < pData->getMaxNumData(); i++)
    {
        findAndUpdateClosestCluster(i);
    }

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(), pLengths->data(),
                   pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_FLOAT, pDistances->data(), pLengths->data(),
                   pDisplacements->data(), MPI_FLOAT, MPI_COMM_WORLD);
}