#include "Utils.hpp"
#include <numeric>

ClosestCluster findClosestCluster(value_t *datapoint, std::vector<value_t> *clusters, const int &numExistingClusters,
                                  const int &numFeatures, IDistanceFunctor *distanceFunc)
{

    int clusterIdx;
    value_t tempDistance, minDistance = -1;

    for (int i = 0; i < numExistingClusters; i++)
    {
        tempDistance = (*distanceFunc)(datapoint, clusters->data() + (i * numFeatures), numFeatures);

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx = i;
        }
    }

    return ClosestCluster{clusterIdx, minDistance};
}

int weightedRandomSelection(const int &maxIdx, std::vector<value_t> *weights, float randomFrac)
{

    value_t cutoff = randomFrac * std::accumulate(weights->begin(), weights->end(), 0);

    // each iteration substract the weight from the cutoff and once it reaches <= 0, the corresponding index is selected
    for (int i = 0; i < maxIdx; i++)
    {
        if ((cutoff -= weights->at(i)) <= 0)
        {
            return i;
        }
    }

    return maxIdx;
}