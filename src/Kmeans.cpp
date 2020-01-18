#include "Kmeans.hpp"
#include <ctime>

void Kmeans::fit(Matrix *matrix, int numClusters, int numRestarts)
{
    std::vector<value_t> weights(matrix->numRows, 1);
    fit(matrix, numClusters, numRestarts, &weights);
}

void Kmeans::fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights)
{
    for (int i = 0; i < numRestarts; i++)
    {
        ClusterData clusterData(matrix->numRows, matrix->numCols, numClusters);
        initializer->setUp(matrix, &clusterData, weights);
        maximizer->setUp(matrix, &clusterData, weights);

        initializer->initialize(distanceFunc, time(NULL) * (float)i);
        auto distances = maximizer->maximize(distanceFunc);

        compareResults(&clusterData, &distances);
    }
}