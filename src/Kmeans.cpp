#include "Kmeans.hpp"
#include "mpi.h"
#include <ctime>
#include <iostream>

void Kmeans::fit(Matrix *matrix, int numClusters, int numRestarts)
{
    std::vector<value_t> weights(matrix->numRows, 1);
    fit(matrix, numClusters, numRestarts, &weights);
}

void Kmeans::fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights)
{
    for (int i = 0; i < numRestarts; i++)
    {
        std::vector<value_t> distances(matrix->numRows, -1);
        ClusterData clusterData(matrix->numRows, matrix->numCols, numClusters);
        BundledAlgorithmData bundledData = {matrix, &clusterData, weights};

        initializer->setUp(&bundledData);
        maximizer->setUp(&bundledData);

        initializer->initialize(&distances, distanceFunc, time(NULL) * (float)i);
        maximizer->maximize(&distances, distanceFunc);

        compareResults(&clusterData, &distances);
    }
}

MPIDataChunks MPIKmeans::initMPIMembers(Matrix *matrix, std::vector<value_t> *weights)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // broadcast the number of rows and columns of matrix to each MPI process' matrix
    MPI_Bcast(&matrix->numRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix->numCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // number of datapoints allocated for each process to compute
    int chunk = matrix->numRows / numProcs;
    int scrap = chunk + (matrix->numRows % numProcs);

    // Size of each sub-array to gather
    std::vector<int> vLens(numProcs);
    // Index of each sub-array to gather
    std::vector<int> vDisps(numProcs);
    for (int i = 0; i < numProcs; i++)
    {
        vLens[i] = chunk;
        vDisps[i] = i * chunk;
    }
    vLens[numProcs - 1] = scrap;

    // Create disp/len arrays for data scatter
    int dataLens[numProcs];
    int dataDisps[numProcs];
    for (int i = 0; i < numProcs; i++)
    {
        dataLens[i] = vLens[i] * matrix->numCols;
        dataDisps[i] = vDisps[i] * matrix->numCols;
    }

    Matrix matrixChunk;
    matrixChunk.data.resize(dataLens[rank]);
    matrixChunk.numRows = vLens[rank];
    matrixChunk.numCols = matrix->numCols;

    MPI_Scatterv(matrix->data.data(), dataLens, dataDisps, MPI_FLOAT, matrixChunk.data.data(), dataLens[rank],
                 MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weights->data(), weights->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    return MPIDataChunks{rank, matrixChunk, vLens, vDisps};
}

void MPIKmeans::fit(Matrix *matrix, int numClusters, int numRestarts)
{
    std::vector<value_t> weights(matrix->numRows, 1);
    fit(matrix, numClusters, numRestarts, &weights);
}

void MPIKmeans::fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights)
{
    auto dataChunks = initMPIMembers(matrix, weights);

    for (int i = 0; i < numRestarts; i++)
    {
        std::vector<value_t> distances(matrix->numRows, -1);
        ClusterData clusterData(matrix->numRows, matrix->numCols, numClusters);
        BundledMPIAlgorithmData bundledData = {matrix, &clusterData, weights, &dataChunks};

        initializer->setUp(&bundledData);
        maximizer->setUp(&bundledData);

        initializer->initialize(&distances, distanceFunc, time(NULL) * (float)i);
        maximizer->maximize(&distances, distanceFunc);

        if (dataChunks.rank == 0)
        {
            compareResults(&clusterData, &distances);
        }
    }
}