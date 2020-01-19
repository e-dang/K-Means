#pragma once

#include "AbstractKmeansAlgorithms.hpp"

/**
 * @brief Implementation of a Kmeans initialization aglorithm. Selects datapoints to be new clusters at random weighted
 *        by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 */
class SerialKPlusPlus : public AbstractKmeansInitializer
{
protected:
    /**
     * @brief Helper method that initializes the first cluster to the datapoint whose index is randIdx, thus randIdx
     *        should be an integer generated uniformly at random in the range of [0, numData).
     *
     * @param randIdx - The index of the datapoint to make as the first cluster, drawn at random.
     */
    void initializeFirstCluster(int randIdx);

    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    void findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc);

    /**
     * @brief Helper function that selects a datapoint to be a new cluster center with a probability proportional to the
     *        square of the distance to its current closest cluster.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param randFrac - A randomly generated float in the range of [0, 1) needed by weightedRandomSelection().
     */
    void weightedClusterSelection(std::vector<value_t> *distances, float randFrac);

public:
    /**
     * @brief Destroy the Serial KPlusPlus object
     */
    ~SerialKPlusPlus(){};

    /**
     * @brief Top level function that initializes the clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @param seed - The seed for the RNG.
     */
    void initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed) override;
};