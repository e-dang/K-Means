#include "Kmeans.hpp"
#include "mpi.h"
#include <ctime>

void Kmeans::fit(Matrix *data, int numClusters, int numRestarts)
{
    std::vector<value_t> weights(data->getMaxNumData(), 1);
    fit(data, numClusters, numRestarts, &weights);
}

void Kmeans::fit(Matrix *data, int numClusters, int numRestarts, std::vector<value_t> *weights)
{
    auto staticData = initStaticData(data, weights);
    initializer->setStaticData(&staticData);
    maximizer->setStaticData(&staticData);

    for (int i = 0; i < numRestarts; i++)
    {
        std::vector<value_t> distances(data->getMaxNumData(), 1);
        ClusterData clusterData(data->getMaxNumData(), data->getNumFeatures(), numClusters);

        initializer->setDynamicData(&clusterData, &distances);
        maximizer->setDynamicData(&clusterData, &distances);

        initializer->initialize(time(NULL) * (float)i);
        maximizer->maximize();

        compareResults(&clusterData, &distances);
    }
}

StaticData MPIKmeans::initStaticData(Matrix *data, std::vector<value_t> *weights)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // number of datapoints allocated for each process to compute
    int chunk = data->getMaxNumData() / numProcs;
    int scrap = chunk + (data->getMaxNumData() % numProcs);

    std::vector<int> lengths(numProcs);       // size of each sub-array to gather
    std::vector<int> displacements(numProcs); // index of each sub-array to gather
    for (int i = 0; i < numProcs; i++)
    {
        lengths[i] = chunk;
        displacements[i] = i * chunk;
    }
    lengths[numProcs - 1] = scrap;

    return StaticData{data,
                      weights,
                      distanceFunc,
                      rank,
                      mTotalNumData,
                      lengths,
                      displacements};
}

void MPIKmeans::fit(Matrix *data, int numClusters, int numRestarts)
{
    std::vector<value_t> weights(data->getMaxNumData(), 1);
    fit(data, numClusters, numRestarts, &weights);
}

void MPIKmeans::fit(Matrix *data, int numClusters, int numRestarts, std::vector<value_t> *weights)
{
    auto staticData = initStaticData(data, weights);

    initializer->setStaticData(&staticData);
    maximizer->setStaticData(&staticData);

    for (int i = 0; i < numRestarts; i++)
    {
        std::vector<value_t> distances(data->getMaxNumData(), 1);
        ClusterData clusterData(data->getMaxNumData(), data->getNumFeatures(), numClusters);

        initializer->setDynamicData(&clusterData, &distances);
        maximizer->setDynamicData(&clusterData, &distances);

        initializer->initialize(time(NULL) * (float)i);
        maximizer->maximize();

        compareResults(&clusterData, &distances);
    }
}