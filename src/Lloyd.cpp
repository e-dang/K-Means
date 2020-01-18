
#include "Lloyd.hpp"
#include "Utils.hpp"
#include <iostream>

std::vector<value_t> SerialLloyd::maximize(IDistanceFunctor *distanceFunc)
{
    int changed;
    std::vector<value_t> distances(matrix->numRows, -1);

    do
    {
        updateClusters();
        changed = reassignPoints(&distances, distanceFunc);

    } while (changed > (matrix->numRows * 0.0001)); // do until 99.9% of data doesnt change

    return distances;
}

void SerialLloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(clusters->data.begin(), clusters->data.end(), 0);

    // calc the sum of each feature for all points belonging to a cluster
    for (int i = 0; i < matrix->numRows; i++)
    {
        for (int j = 0; j < matrix->numCols; j++)
        {
            clusters->at(clustering->at(i), j) += matrix->at(i, j);
        }
    }

    // average out the sum of each cluster based on the number of datapoints assigned to it
    for (int i = 0; i < clusters->numRows; i++)
    {
        for (int j = 0; j < clusters->numCols; j++)
        {
            clusters->data.at(i * clusters->numCols + j) /= clusterCounts->at(i);
        }
    }
}

int SerialLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < matrix->numRows; i++)
    {
        int before = clustering->at(i);

        // find closest cluster for each datapoint and update cluster assignment
        auto closestCluster = findClosestCluster(&*(matrix->at(i)), &clusters->data, clusters->numRows,
                                                 matrix->numCols, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != clustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}