
#include "KmeansAlgorithms/Lloyd.hpp"

#include "mpi.h"

void TemplateLloyd::maximize()
{
    int_fast32_t changed, minNumChanged = (*pTotalNumData * MIN_PERCENT_CHANGED);

    do
    {
        (*ppClusters)->fill(0);

        calcClusterSums();

        averageClusterSums();

        changed = reassignPoints();

    } while (changed > minNumChanged);
}

void SharedMemoryLloyd::calcClusterSums() { pAverager->calculateSum(pData, *ppClusters, *ppClustering, pWeights); }

void SharedMemoryLloyd::averageClusterSums() { pAverager->normalizeSum(*ppClusters, *ppClusterWeights); }

int_fast32_t SharedMemoryLloyd::reassignPoints() { return pPointReassigner->reassignPoints(pKmeansData); }

void MPILloyd::calcClusterSums()
{
    pAverager->calculateSum(pData, *ppClusters, *ppClustering, pWeights, pDisplacements->at(*pRank));

    MPI_Allreduce(MPI_IN_PLACE, (*ppClusters)->data(), (*ppClusters)->size(), mpi_type_t, MPI_SUM, MPI_COMM_WORLD);
}

void MPILloyd::averageClusterSums()
{
    std::vector<value_t> copyWeights((*ppClusterWeights)->size());
    MPI_Reduce((*ppClusterWeights)->data(), copyWeights.data(), copyWeights.size(), mpi_type_t, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (*pRank == 0)
    {
        pAverager->normalizeSum(*ppClusters, &copyWeights);
    }

    MPI_Bcast((*ppClusters)->data(), (*ppClusters)->size(), mpi_type_t, 0, MPI_COMM_WORLD);
}

int_fast32_t MPILloyd::reassignPoints()
{
    int_fast32_t changed = pPointReassigner->reassignPoints(pKmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(*pRank), MPI_INT, (*ppClustering)->data(), pLengths->data(),
                   pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(*pRank), mpi_type_t, (*ppSqDistances)->data(), pLengths->data(),
                   pDisplacements->data(), mpi_type_t, MPI_COMM_WORLD);

    return changed;
}