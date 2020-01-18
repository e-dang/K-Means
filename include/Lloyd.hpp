#pragma once

#include "AbstractKmeansAlgorithms.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Implementation of a Kmeans maximization algorithm. Given a set of initialized clusters, this class will
 *        optimize the clusters using Lloyd's algorithm.
 */
class SerialLloyd : public AbstractKmeansMaximizer
{
public:
    /**
     * @brief Top level function for running Lloyd's algorithm on a set of pre-initialized clusters.
     *
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @return std::vector<value_t> - A vector of the squared distances of every point to its closest cluster.
     */
    std::vector<value_t> maximize(IDistanceFunctor *distanceFunc) override;

    /**
     * @brief Helper function that updates clusters based on the center of mass of the points assigned to it.
     */
    void updateClusters();

    /**
     * @brief Helper function that checks if each point's closest cluster has changed after the clusters have been
     *        updated in the call to updateClusters(), and updates the clustering data if necessary. This function also
     *        keeps track of the number of datapoints that have changed cluster assignments and returns this value.
     *
     * @param distances - A vector to store the square distances of each point to their assigned cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @return int - The number of datapoints whose cluster assignment has changed in the current iteration.
     */
    int reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc);
};