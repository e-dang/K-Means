#include "Kmeans.hpp"

Kmeans::Kmeans(int numClusters, int numRestarts) : numClusters(numClusters), numRestarts(numRestarts)
{
    bestError = INT_MAX;
}

Kmeans::~Kmeans()
{
}

bool Kmeans::setNumClusters(int numClusters)
{
    this->numClusters = numClusters;
    return true;
}

bool Kmeans::setNumRestarts(int numRestarts)
{
    this->numRestarts = numRestarts;
    return true;
}