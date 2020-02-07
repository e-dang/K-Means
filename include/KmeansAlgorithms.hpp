#pragma once

#include <memory>

#include "ClosestClusterUpdater.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "DistanceFunctors.hpp"
#include "PointReassigner.hpp"

/**
 * @brief Abstract class that all Kmeans algorithms, such as initializers and maximizers will derive from. This class
 *        contains code that is used to set up each of these algorithms.
 */
class AbstractKmeansAlgorithm
{
protected:
    KmeansData* pKmeansData;  // useful for passing data to strategy pattern algorithms

    // user data
    const Matrix* pData;
    const std::vector<value_t>* pWeights;
    std::shared_ptr<IDistanceFunctor> pDistanceFunc;

    // cluster data
    Matrix** ppClusters;
    std::vector<int_fast32_t>** ppClustering;
    std::vector<value_t>** ppClusterWeights;
    std::vector<value_t>** ppSqDistances;

    // chunk data
    const int* pRank;
    const int_fast32_t* pTotalNumData;
    const std::vector<int_fast32_t>* pLengths;
    const std::vector<int_fast32_t>* pDisplacements;

public:
    AbstractKmeansAlgorithm(){};

    virtual ~AbstractKmeansAlgorithm(){};

    void setKmeansData(KmeansData* kmeansData)
    {
        pKmeansData      = kmeansData;
        pData            = kmeansData->pData;
        pWeights         = kmeansData->pWeights;
        ppClusters       = &kmeansData->pClusters;
        ppClustering     = &kmeansData->pClustering;
        ppClusterWeights = &kmeansData->pClusterWeights;
        ppSqDistances    = &kmeansData->pSqDistances;
        pLengths         = &kmeansData->mLengths;
        pRank            = &kmeansData->mRank;
        pTotalNumData    = &kmeansData->mTotalNumData;
        pDisplacements   = &kmeansData->mDisplacements;
        pDistanceFunc    = kmeansData->pDistanceFunc;
    }
};

/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
class AbstractKmeansInitializer : public AbstractKmeansAlgorithm
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater> pUpdater;

public:
    AbstractKmeansInitializer(AbstractClosestClusterUpdater* updater) : pUpdater(updater) {}

    virtual ~AbstractKmeansInitializer(){};

    /**
     * @brief Interface that Kmeans initialization algorithms must follow for initializing the clusters.
     */
    virtual void initialize() = 0;
};

/**
 * @brief Abstract class that defines the interface for Kmeans maximization algorithms, such as Lloyd's algorithm.
 */
class AbstractKmeansMaximizer : public AbstractKmeansAlgorithm
{
protected:
    const double MIN_PERCENT_CHANGED = 0.0001;  // the % amount of data points allowed to changed before going to next
                                                // iteration

    std::unique_ptr<AbstractPointReassigner> pPointReassigner;

public:
    AbstractKmeansMaximizer(AbstractPointReassigner* pointReassigner) : pPointReassigner(pointReassigner){};

    virtual ~AbstractKmeansMaximizer(){};

    /**
     * @brief Interface that Kmeans maximization algorithms must follow for finding the best clustering given a set of
     *        pre-initialized clusters.
     */
    virtual void maximize() = 0;
};
