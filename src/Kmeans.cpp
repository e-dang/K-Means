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
        std::vector<value_t> distances(matrix->numRows, -1);
        ClusterData clusterData(matrix->numRows, matrix->numCols, numClusters);

        initializer->setUp(matrix, &clusterData, weights);
        maximizer->setUp(matrix, &clusterData, weights);

        initializer->initialize(&distances, distanceFunc, time(NULL) * (float)i);
        maximizer->maximize(&distances, distanceFunc);

        compareResults(&clusterData, &distances);
    }
}