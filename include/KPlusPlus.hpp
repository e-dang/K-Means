#pragma once

#include "AbstractKmeansAlgorithms.hpp"
#include "RandomSelector.hpp"
#include "ClosestClusterFinder.hpp"
#include "ClusteringUpdater.hpp"
#include <memory>

/**
 * @brief Implementation of a Kmeans++ initialization aglorithm. Selects datapoints to be new clusters at random
 *        weighted by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 */
class TemplateKPlusPlus : public AbstractKmeansInitializer
{
protected:
    // Member variables
    std::unique_ptr<AbstractWeightedRandomSelector> pSelector;
    std::unique_ptr<AbstractClosestClusterFinder> pFinder;
    std::unique_ptr<AbstractClusteringUpdater> pUpdater;

public:
    /**
     * @brief Construct a new TemplateKPlusPlus object
     *
     * @param selector
     * @param finder
     */
    TemplateKPlusPlus(AbstractWeightedRandomSelector *selector, AbstractClosestClusterFinder *finder,
                      AbstractClusteringUpdater *updater) : pSelector(selector),
                                                            pFinder(finder),
                                                            pUpdater(updater){};

    /**
     * @brief Destroy the Serial KPlusPlus object
     */
    virtual ~TemplateKPlusPlus(){};

    /**
     * @brief Template function that initializes the clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @param seed - The seed for the RNG.
     */
    void initialize(const float &seed) override;

protected:
    /**
     * @brief Helper function that selects a datapoint to be a new cluster center with a probability proportional to the
     *        square of the distance to its current closest cluster.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param randFrac - A randomly generated float in the range of [0, 1) needed by weightedRandomSelection().
     */
    virtual void weightedClusterSelection(float randFrac);

    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    virtual void findAndUpdateClosestCluster();
};

class KPlusPlus : public TemplateKPlusPlus
{
public:
    KPlusPlus() : TemplateKPlusPlus(new SingleWeightedRandomSelector(), new ClosestClusterFinder(&pClusters),
                                    new ClusteringUpdater(&pClustering, &pClusterWeights)){};
    virtual ~KPlusPlus(){};
};

// /**
//  * @brief Optimized version of KPlusPlus that only differs in the implementation of findAndUpdateClosestCluster(). The
//  *        optimization made to the K++ algorithm is noticing that you don't need to recalculate the distances between
//  *        each point and each cluster each time a cluster is added. Rather you can calculate the distance between each
//  *        point and the newly added cluster each iteration because up until then the datapoint is already assigned to
//  *        its closest cluster out of all existing clusters. Thus we need only to compare that distance to the distance
//  *        between the datapoint and the newly added cluster and update if necessary.
//  */
// class OptimizedKPlusPlus : public TemplateKPlusPlus
// {
// public:
//     OptimizedKPlusPlus() : TemplateKPlusPlus(new WeightedClusterSelection(this), new OptFindAndUpdateClosestCluster(this)){};

//     /**
//      * @brief Destroy the OptimizedKPlusPlus object
//      */
//     virtual ~OptimizedKPlusPlus(){};
// };

/**
 * @brief Parallelized version of the KPlusPlus algorithm using OMP thread parallelism in findAndUpdateClosestCluster().
 *        To change the number of threads, use the environment variable OMP_NUM_THREADS.
 */
class OMPKPlusPlus : public TemplateKPlusPlus
{
public:
    OMPKPlusPlus() : TemplateKPlusPlus(new SingleWeightedRandomSelector(), new ClosestClusterFinder(&pClusters),
                                       new ClusteringUpdater(&pClustering, &pClusterWeights)){};
    // OMPKPlusPlus(AbstractWeightedClusterSelection *selector, AbstractFindAndUpdate *finder) : TemplateKPlusPlus(selector, finder){};

    /**
     * @brief Destroy the OMPKPlusPlus object
     *
     */
    virtual ~OMPKPlusPlus(){};

protected:
    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A functor that defines the distance metric.
     */
    void findAndUpdateClosestCluster() override;
};

// /**
//  * @brief Parallelized version of the OptimizedKPlusPlus algorithm using OMP thread parallelism in
//  *        findAndUpdateClosestCluster(). To change the number of threads, use the environment variable OMP_NUM_THREADS.
//  */
// class OMPOptimizedKPlusPlus : public OMPKPlusPlus
// {
// public:
//     OMPOptimizedKPlusPlus() : OMPKPlusPlus(new WeightedClusterSelection(this), new OptFindAndUpdateClosestCluster(this)){};

//     /**
//      * @brief Destroy the OptimizedKPlusPlus object
//      */
//     virtual ~OMPOptimizedKPlusPlus(){};
// };

// class MPIKPlusPlus : public TemplateKPlusPlus
// {
// public:
//     MPIKPlusPlus() : TemplateKPlusPlus(new WeightedClusterSelection(this), new FindAndUpdateClosestCluster(this)){};
//     MPIKPlusPlus(AbstractWeightedClusterSelection *selector, AbstractFindAndUpdate *finder) : TemplateKPlusPlus(
//                                                                                                   selector, finder){};
//     virtual ~MPIKPlusPlus(){};

// protected:
//     /**
//      * @brief Helper function that selects a datapoint to be a new cluster center with a probability proportional to the
//      *        square of the distance to its current closest cluster.
//      *
//      * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
//      *                    cluster.
//      * @param randFrac - A randomly generated float in the range of [0, 1) needed by weightedRandomSelection().
//      */
//     void weightedClusterSelection(Matrix *clusters, std::vector<value_t> *distances, float randFrac) override;

//     /**
//      * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
//      *        find the closest cluster for each datapoint and update the clustering assignments.
//      *
//      * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
//      *                    cluster.
//      * @param distanceFunc - A functor that defines the distance metric.
//      */
//     void findAndUpdateClosestCluster(std::vector<int> *clustering, std::vector<value_t> *clusterWeights,
//                                      std::vector<value_t> *distances) override;
// };

// class MPIOptimizedKPlusPlus : public MPIKPlusPlus
// {
// public:
//     MPIOptimizedKPlusPlus() : MPIKPlusPlus(new WeightedClusterSelection(this), new OptFindAndUpdateClosestCluster(this)){};
//     virtual ~MPIOptimizedKPlusPlus(){};
// };