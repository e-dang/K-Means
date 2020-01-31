#pragma once

#include "Definitions.hpp"
#include "DataClasses.hpp"
#include "DistanceFunctors.hpp"
#include "ClosestClusterFinder.hpp"
#include "ClusteringUpdater.hpp"
#include <memory>

/**
 * @brief Abstract class that all Kmeans algorithms, such as initializers and maximizers will derive from. This class
 *        contains code that is used to set up each of these algorithms.
 */
class AbstractKmeansAlgorithm
{
protected:
    // user data
    Matrix *pData;
    std::vector<value_t> *pWeights;
    IDistanceFunctor *pDistanceFunc;

    // cluster data
    Matrix *pClusters;
    std::vector<int> *pClustering;
    std::vector<value_t> *pClusterWeights;
    std::vector<value_t> *pDistances;

    // chunk data
    int mRank;
    int mTotalNumData;
    std::vector<int> *pLengths;
    std::vector<int> *pDisplacements;

    // algorithms
    std::unique_ptr<AbstractClosestClusterFinder> pFinder;
    std::unique_ptr<AbstractClusteringUpdater> pUpdater;

public:
    AbstractKmeansAlgorithm() {}

    AbstractKmeansAlgorithm(AbstractClosestClusterFinder *finder,
                            AbstractClusteringUpdater *updater) : pFinder(finder), pUpdater(updater) {}
    /**
     * @brief Destroy the AbstractKmeansAlgorithm object
     */
    virtual ~AbstractKmeansAlgorithm(){};

    void setStaticData(StaticData *staticData)
    {
        pData = staticData->pData;
        pWeights = staticData->pWeights;
        pLengths = &staticData->mLengths;
        pDisplacements = &staticData->mDisplacements;
        pDistanceFunc = staticData->pDistanceFunc;
        mRank = staticData->mRank;
        mTotalNumData = staticData->mTotalNumData;
    }

    void setDynamicData(ClusterData *clusterData, std::vector<value_t> *distances)
    {
        pClustering = &clusterData->mClustering;
        pClusters = &clusterData->mClusters;
        pClusterWeights = &clusterData->mClusterWeights;
        pDistances = distances;
    }

protected:
    void findAndUpdateClosestCluster(const int &dataIdx)
    {
        int displacedDataIdx = pDisplacements->at(mRank) + dataIdx;

        auto closestCluster = pFinder->findClosestCluster(pData->at(dataIdx), pDistanceFunc);
        if (pDistances->at(displacedDataIdx) > closestCluster.distance || pDistances->at(displacedDataIdx) < 0)
        {
            pUpdater->update(displacedDataIdx, closestCluster.clusterIdx, pWeights->at(dataIdx));
            pDistances->at(displacedDataIdx) = closestCluster.distance;
        }
    }
};

/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
class AbstractKmeansInitializer : public AbstractKmeansAlgorithm
{
public:
    AbstractKmeansInitializer(AbstractClosestClusterFinder *finder,
                              AbstractClusteringUpdater *updater) : AbstractKmeansAlgorithm(finder, updater) {}
    /**
     * @brief Destroy the AbstractKmeansInitializer object
     */
    virtual ~AbstractKmeansInitializer(){};

    /**
     * @brief Interface that Kmeans initialization algorithms must follow for initializing the clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A pointer to a class that calculates distances between points and is an implementation of
     *                       IDistanceFunctor.
     * @param seed - The number to seed the RNG.
     */
    virtual void initialize(const float &seed) = 0;
};

/**
 * @brief Abstract class that defines the interface for Kmeans maximization algorithms, such as Lloyd's algorithm.
 */
class AbstractKmeansMaximizer : public AbstractKmeansAlgorithm
{
protected:
    // Constants
    const float MIN_PERCENT_CHANGED = 0.0001; // the % amount of data points allowed to changed before going to next
                                              // iteration
public:
    AbstractKmeansMaximizer(AbstractClosestClusterFinder *finder,
                            AbstractClusteringUpdater *updater) : AbstractKmeansAlgorithm(finder, updater) {}
    /**
     * @brief Destroy the AbstractKmeansMaximizer object
     */
    virtual ~AbstractKmeansMaximizer(){};

    /**
     * @brief Interface that Kmeans maximization algorithms must follow for finding the best clustering given a set of
     *        pre-initialized clusters.
     *
     * @param distances - A pointer to a vector that stores the squared distances of each datapoint to its closest
     *                    cluster.
     * @param distanceFunc - A pointer to a class that calculates distances between points and is an implementation of
     *                       IDistanceFunctor.
     */
    virtual void maximize() = 0;
};
