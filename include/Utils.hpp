#pragma once

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Helper function that find the closest cluster and corresponding distance for a given datapoint.
 *
 * @param datapoint - A pointer to the first element of the datapoint.
 * @param clusters - A pointer to the vector containing the clusters.
 * @param numExistingClusters - The number of clusters contained in the vector.
 * @param numFeatures - The number of features in the datapoint.
 * @param distanceFunc - A functor that defines the distance metric.
 * @return ClosestCluster - struct containing the cluster index of the closest cluster and the corresponding distance.
 */
ClosestCluster findClosestCluster(value_t *datapoint, std::vector<value_t> *clusters, const int &numExistingClusters,
                                  const int &numFeatures, IDistanceFunctor *distanceFunc);

/**
 * @brief Algorithm for a weighted random selection of an index in the range of [0, maxIdx).
 *
 * @param maxIdx - The number that defines the maximum range of indexes to select from. This number is non-inclusive.
 * @param weights - The weights of the indices to select from.
 * @param partialWeightSum - A random float used in the algorithm to perform the random selection.
 * @return int - The selected index.
 */
int weightedRandomSelection(const int &maxIdx, std::vector<value_t> *weights, float partialWeightSum);