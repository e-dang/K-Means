#include "Strategies/CoresetClusteringFinisher.hpp"

#include <numeric>

#include "mpi.h"

value_t SharedMemoryCoresetClusteringFinisher::finishClustering(KmeansData* const kmeansData)
{
    pUpdater->findAndUpdateClosestClusters(kmeansData);

    return std::accumulate(kmeansData->pSqDistances->begin(), kmeansData->pSqDistances->end(), 0.0);
}

value_t MPICoresetClusteringFinisher::finishClustering(KmeansData* const kmeansData)
{
    pUpdater->findAndUpdateClosestClusters(kmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansData->mLengths.at(kmeansData->mRank), MPI_INT, kmeansData->pClustering->data(),
                   kmeansData->mLengths.data(), kmeansData->mDisplacements.data(), MPI_INT, MPI_COMM_WORLD);

    MPI_Allgatherv(MPI_IN_PLACE, kmeansData->mLengths.at(kmeansData->mRank), mpi_type_t,
                   kmeansData->pSqDistances->data(), kmeansData->mLengths.data(), kmeansData->mDisplacements.data(),
                   mpi_type_t, MPI_COMM_WORLD);

    return std::accumulate(kmeansData->pSqDistances->begin(), kmeansData->pSqDistances->end(), 0.0);
}