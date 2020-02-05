#include "KPlusPlus.hpp"

#include <omp.h>

#include "Utils.hpp"
#include "mpi.h"

void TemplateKPlusPlus::initialize()
{
    // initialize first cluster uniformly at random. Thus distances should be filled with same number i.e. 1
    weightedClusterSelection();

    // change fill distances vector with -1 so values aren't confused with actual distances
    std::fill((*ppSqDistances)->begin(), (*ppSqDistances)->end(), -1);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int i = 1; i < (*ppClusters)->getMaxNumData(); i++)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestClusters();

        // select point to be next cluster center weighted by nearest distance squared
        weightedClusterSelection();
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestClusters();
}

void SharedMemoryKPlusPlus::weightedClusterSelection()
{
    value_t randSumFrac = getRandDouble01() * std::accumulate((*ppSqDistances)->begin(), (*ppSqDistances)->end(), 0.0);
    int dataIdx         = pSelector->select(*ppSqDistances, randSumFrac);
    (*ppClusters)->appendDataPoint(pData->at(dataIdx));
}

void SharedMemoryKPlusPlus::findAndUpdateClosestClusters() { pUpdater->findAndUpdateClosestClusters(pKmeansData); }

void MPIKPlusPlus::weightedClusterSelection()
{
    int dataIdx;
    if (*pRank == 0)
    {
        value_t randSumFrac =
          getRandDouble01MPI() * std::accumulate((*ppSqDistances)->begin(), (*ppSqDistances)->end(), 0.0);
        dataIdx = pSelector->select((*ppSqDistances), randSumFrac);
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

    if (*pRank == rank)
    {
        (*ppClusters)->appendDataPoint(pData->at(dataIdx - pDisplacements->at(*pRank)));
    }
    else
    {
        (*ppClusters)->reserve((*ppClusters)->getNumData() + 1);
    }

    MPI_Bcast((*ppClustering)->data(), (*ppClustering)->size(), MPI_INT, rank, MPI_COMM_WORLD);
    MPI_Bcast((*ppClusters)->data(), (*ppClusters)->size(), MPI_FLOAT, rank, MPI_COMM_WORLD);
}

void MPIKPlusPlus::findAndUpdateClosestClusters()
{
    pUpdater->findAndUpdateClosestClusters(pKmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(*pRank), MPI_INT, (*ppClustering)->data(), pLengths->data(),
                   pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(*pRank), MPI_FLOAT, (*ppSqDistances)->data(), pLengths->data(),
                   pDisplacements->data(), MPI_FLOAT, MPI_COMM_WORLD);
}