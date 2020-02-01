#include "Kmeans.hpp"

#include <ctime>

#include "mpi.h"

ClusterResults AbstractKmeans::run(Matrix* data, const int& numClusters, const int& numRestarts, StaticData* staticData)
{
    ClusterResults clusterResults;

    pInitializer->setStaticData(staticData);
    pMaximizer->setStaticData(staticData);

    for (int i = 0; i < numRestarts; i++)
    {
        std::vector<value_t> distances(staticData->mTotalNumData, 1);
        ClusterData clusterData(staticData->mTotalNumData, data->getNumFeatures(), numClusters);

        pInitializer->setDynamicData(&clusterData, &distances);
        pMaximizer->setDynamicData(&clusterData, &distances);

        pInitializer->initialize(time(NULL) * (float)i);
        pMaximizer->maximize();

        compareResults(&clusterData, &distances, &clusterResults);
    }

    return clusterResults;
}

ClusterResults AbstractKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts)
{
    std::vector<value_t> weights(data->getMaxNumData(), 1);
    return fit(data, numClusters, numRestarts, &weights);
}

ClusterResults AbstractKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                   std::vector<value_t>* weights)
{
    auto staticData = initStaticData(data, weights);
    return run(data, numClusters, numRestarts, &staticData);
}

StaticData MPIKmeans::initStaticData(Matrix* data, std::vector<value_t>* weights)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // number of datapoints allocated for each process to compute
    int chunk = mTotalNumData / numProcs;
    int scrap = chunk + (mTotalNumData % numProcs);

    std::vector<int> lengths(numProcs);        // size of each sub-array to gather
    std::vector<int> displacements(numProcs);  // index of each sub-array to gather
    for (int i = 0; i < numProcs; i++)
    {
        lengths[i]       = chunk;
        displacements[i] = i * chunk;
    }
    lengths[numProcs - 1] = scrap;

    return StaticData{ data, weights, pDistanceFunc, rank, mTotalNumData, lengths, displacements };
}