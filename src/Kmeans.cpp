#include "Kmeans.hpp"

Kmeans::Kmeans(int numClusters, int numRestarts) : numClusters(numClusters), numRestarts(numRestarts)
{
    finalError = INT_MAX;
}

Kmeans::~Kmeans()
{
}