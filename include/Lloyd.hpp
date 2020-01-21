#pragma once

#include "AbstractKmeansAlgorithms.hpp"
#include "DistanceFunctors.hpp"
#include <memory>

/**
 * @brief Implementation of a Kmeans maximization algorithm. Given a set of initialized clusters, this class will
 *        optimize the clusters using Lloyd's algorithm.
 */
class TemplateLloyd : public AbstractKmeansMaximizer
{
protected:
    std::unique_ptr<AbstractFindAndUpdate> pFinder;

public:
    TemplateLloyd(AbstractFindAndUpdate *finder) : pFinder(finder){};

    /**
     * @brief Destroy the SerialLloyd object
     */
    virtual ~TemplateLloyd(){};

    /**
     * @brief Top level function for running Lloyd's algorithm on a set of pre-initialized clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     */
    void maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;

protected:
    /**
     * @brief Helper function that updates clusters based on the center of mass of the points assigned to it.
     */
    virtual void calcClusterSums();

    virtual void averageClusterSums();

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
};

class Lloyd : public TemplateLloyd
{
public:
    Lloyd() : TemplateLloyd(new FindAndUpdateClosestCluster(this)){};
    ~Lloyd(){};
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
class OptimizedLloyd : public TemplateLloyd
{
public:
    OptimizedLloyd() : TemplateLloyd(new FindAndUpdateClosestCluster(this)){};

    /**
     * @brief Destroy the OptimizedSerialLloyd object
     */
    ~OptimizedLloyd(){};

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

/**
 * @brief Parallelized version of Lloyd's algorithm using OMP thread parallelism in both updateClusters() and
 *        reassignPoints(). To change the number of threads, use the environment variable OMP_NUM_THREADS.
 */
class OMPLloyd : public TemplateLloyd, public AbstractOMPKmeansAlgorithm
{
public:
    OMPLloyd() : TemplateLloyd(new FindAndUpdateClosestCluster(this)){};
    OMPLloyd(AbstractFindAndUpdate *finder) : TemplateLloyd(finder){};

    /**
     * @brief Destroy the OMPLloyd object
     *
     */
    virtual ~OMPLloyd(){};

protected:
    void averageClusterSums() override;

    /**
     * @brief Helper function that checks if each point's closest cluster has changed after the clusters have been
     *        updated in the call to updateClusters(), and updates the clustering data if necessary. This function also
     *        keeps track of the number of datapoints that have changed cluster assignments and returns this value.
     *
     * @param distances - A vector to store the square distances of each point to their assigned cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @return int - The number of datapoints whose cluster assignment has changed in the current iteration.
     */
    virtual int reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;
};

/**
 * @brief Parallelized version of the OptimizedLloyd algorithm using OMP thread parallelism. This class has its own
 *        implementation of reassignPoints() but uses OMPLloyd's versions of updateClusters(). To change the  number of
 *        threads, use the environment variable OMP_NUM_THREADS.
 */
class OMPOptimizedLloyd : public OMPLloyd
{
public:
    OMPOptimizedLloyd() : OMPLloyd(new FindAndUpdateClosestCluster(this)){};
    /**
     * @brief Destroy the OptimizedSerialLloyd object
     */
    ~OMPOptimizedLloyd(){};

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

class MPILloyd : public TemplateLloyd, public AbstractMPIKmeansAlgorithm
{
public:
    MPILloyd() : TemplateLloyd(new FindAndUpdateClosestCluster(this)){};
    ~MPILloyd(){};

protected:
    void calcClusterSums() override;
    void averageClusterSums() override;
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