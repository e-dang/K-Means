#pragma once

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Helper function that find the closest cluster and corresponding distance for a given datapoint.
 *
 * @param datapoint - A pointer to the first element of the datapoint.
 * @param clusters - A pointer to the Matrix containing the clusters.
 * @param distanceFunc - A functor that defines the distance metric.
 * @return ClosestCluster - struct containing the cluster index of the closest cluster and the corresponding distance.
 */
ClosestCluster findClosestCluster(value_t *datapoint, Matrix *clusters, IDistanceFunctor *distanceFunc);

/**
 * @brief Algorithm for a weighted random selection of an index in the range of [0, maxIdx), where maxIdx is the length
 *        of the weights vector. The algorithm works by summing the weights and multiplying the result by a random float
 *        that is [0, 1). Then we sequentially subtract each weight from the random sum fraction until the value is less
 *        than or equal to 0 at which point the index of the weight that turned the sum for positive to <= 0 is
 *        returned.
 *
 * @param weights - The weights of the indices to select from.
 * @param randomFrac - A random float used in the algorithm to perform the random selection, [0, 1).
 * @return int - The selected index.
 */
int weightedRandomSelection(std::vector<value_t> *weights, float randomFrac);