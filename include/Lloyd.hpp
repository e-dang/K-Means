#pragma once

#include "AbstractKmeansAlgorithms.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Implementation of a Kmeans maximization algorithm. Given a set of initialized clusters, this class will
 *        optimize the clusters using Lloyd's algorithm.
 */
class SerialLloyd : public AbstractKmeansMaximizer
{
protected:
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
    virtual int reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc);

public:
    /**
     * @brief Destroy the SerialLloyd object
     */
    virtual ~SerialLloyd(){};

    /**
     * @brief Top level function for running Lloyd's algorithm on a set of pre-initialized clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     */
    void maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;
};

/**
 * @brief Optimized version of Lloyd's algorithm that only differs in its choice of when it needs to calculate the
 *        distance between a point and each cluster in order to find which cluster is the closest. The optimization is
 *        testing whether the point has gotten closer to its previously closest cluster, if so then there is no need to
 *        recalculate the distances between that point and every other cluster. If the distance has gotten larger then
 *        it must recalculate the distances to each cluster. This optimization is taken from the paper:
 *        https://link.springer.com/content/pdf/10.1631%2Fjzus.2006.A1626.pdf
 *
 */
class OptimizedSerialLloyd : public SerialLloyd
{
protected:
    /**
     * @brief Helper function that checks if each point's closest cluster has changed after the clusters have been
     *        updated in the call to updateClusters(), and updates the clustering data if necessary. This function also
     *        keeps track of the number of datapoints that have changed cluster assignments and returns this value.
     *
     * @param distances - A vector to store the square distances of each point to their assigned cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @return int - The number of datapoints whose cluster assignment has changed in the current iteration.
     */
    int reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;
};