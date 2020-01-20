#pragma once

#include "AbstractKmeansAlgorithms.hpp"

/**
 * @brief Implementation of a Kmeans++ initialization aglorithm. Selects datapoints to be new clusters at random
 *        weighted by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 */
class KPlusPlus : public AbstractKmeansInitializer
{
protected:
    /**
     * @brief Helper method that initializes the first cluster to the datapoint whose index is randIdx, thus randIdx
     *        should be an integer generated uniformly at random in the range of [0, numData).
     *
     * @param randIdx - The index of the datapoint to make as the first cluster, drawn at random.
     */
    virtual void initializeFirstCluster(int randIdx);

    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    virtual void findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc);

    /**
     * @brief Helper function that selects a datapoint to be a new cluster center with a probability proportional to the
     *        square of the distance to its current closest cluster.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param randFrac - A randomly generated float in the range of [0, 1) needed by weightedRandomSelection().
     */
    virtual void weightedClusterSelection(std::vector<value_t> *distances, float randFrac);

public:
    /**
     * @brief Destroy the Serial KPlusPlus object
     */
    virtual ~KPlusPlus(){};

    /**
     * @brief Top level function that initializes the clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @param seed - The seed for the RNG.
     */
    virtual void initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed) override;
};

/**
 * @brief Optimized version of KPlusPlus that only differs in the implementation of findAndUpdateClosestCluster(). The
 *        optimization made to the K++ algorithm is noticing that you don't need to recalculate the distances between
 *        each point and each cluster each time a cluster is added. Rather you can calculate the distance between each
 *        point and the newly added cluster each iteration because up until then the datapoint is already assigned to
 *        its closest cluster out of all existing clusters. Thus we need only to compare that distance to the distance
 *        between the datapoint and the newly added cluster and update if necessary.
 */
class OptimizedKPlusPlus : public KPlusPlus
{
protected:
    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    void findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;

public:
    /**
     * @brief Destroy the OptimizedKPlusPlus object
     */
    ~OptimizedKPlusPlus(){};
};

/**
 * @brief Parallelized version of the KPlusPlus algorithm using OMP thread parallelism in findAndUpdateClosestCluster().
 *        To change the number of threads, use the environment variable OMP_NUM_THREADS.
 */
class OMPKPlusPlus : public KPlusPlus, public AbstractOMPKmeansAlgorithm
{
protected:
    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    void findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;

public:
    /**
     * @brief Destroy the OMPKPlusPlus object
     *
     */
    virtual ~OMPKPlusPlus(){};
};

/**
 * @brief Parallelized version of the OptimizedKPlusPlus algorithm using OMP thread parallelism in
 *        findAndUpdateClosestCluster(). To change the number of threads, use the environment variable OMP_NUM_THREADS.
 */
class OMPOptimizedKPlusPlus : public OMPKPlusPlus
{
protected:
    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    void findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;

public:
    /**
     * @brief Destroy the OptimizedKPlusPlus object
     */
    ~OMPOptimizedKPlusPlus(){};
};

class MPIKPlusPlus : public KPlusPlus, public AbstractMPIKmeansAlgorithm
{
protected:
    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    void findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;

    /**
     * @brief Helper function that selects a datapoint to be a new cluster center with a probability proportional to the
     *        square of the distance to its current closest cluster.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param randFrac - A randomly generated float in the range of [0, 1) needed by weightedRandomSelection().
     */
    void weightedClusterSelection(std::vector<value_t> *distances, float randFrac) override;
    void initializeFirstCluster(int randIdx) override;

public:
    void initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed) override;

    void setUp(BundledAlgorithmData *bundledData) override;
};
