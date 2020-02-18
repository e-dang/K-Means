#pragma once

#include <memory>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"
#include "Strategies/PointReassigner.hpp"
#include "Utils/DistanceFunctors.hpp"

namespace HPKmeans
{
/**
 * @brief Abstract class that all Kmeans algorithms, such as initializers and maximizers will derive from. This class
 *        contains code that is used to set up each of these algorithms.
 */
template <typename precision, typename int_size>
class AbstractKmeansAlgorithm
{
protected:
    KmeansData<precision, int_size>* pKmeansData;

    // user data
    // const Matrix<precision, int_size>* pData;
    // const std::vector<precision>* pWeights;
    // std::shared_ptr<IDistanceFunctor<precision>> pDistanceFunc;

    // // cluster data
    // Matrix<precision, int_size>** ppClusters;
    // std::vector<int_size>** ppClustering;
    // std::vector<precision>** ppClusterWeights;
    // std::vector<precision>** ppSqDistances;

    // // chunk data
    // const int* pRank;
    // const int_size* pTotalNumData;
    // const std::vector<int_size>* pLengths;
    // const std::vector<int_size>* pDisplacements;

public:
    AbstractKmeansAlgorithm() = default;

    virtual ~AbstractKmeansAlgorithm() = default;

    void setKmeansData(KmeansData<precision, int_size>* kmeansData);
};

/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
template <typename precision, typename int_size>
class AbstractKmeansInitializer : public AbstractKmeansAlgorithm<precision, int_size>
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> pUpdater;

public:
    AbstractKmeansInitializer(AbstractClosestClusterUpdater<precision, int_size>* updater) : pUpdater(updater) {}

    virtual ~AbstractKmeansInitializer() = default;

    /**
     * @brief Interface that Kmeans initialization algorithms must follow for initializing the clusters.
     */
    virtual void initialize() = 0;
};

/**
 * @brief Abstract class that defines the interface for Kmeans maximization algorithms, such as Lloyd's algorithm.
 */
template <typename precision, typename int_size>
class AbstractKmeansMaximizer : public AbstractKmeansAlgorithm<precision, int_size>
{
protected:
    const precision MIN_PERCENT_CHANGED = 0.0001;  // the % amount of data points allowed to changed before going to
                                                   // next iteration

    std::unique_ptr<AbstractPointReassigner<precision, int_size>> pPointReassigner;

public:
    AbstractKmeansMaximizer(AbstractPointReassigner<precision, int_size>* pointReassigner) :
        pPointReassigner(pointReassigner){};

    virtual ~AbstractKmeansMaximizer() = default;

    /**
     * @brief Interface that Kmeans maximization algorithms must follow for finding the best clustering given a set of
     *        pre-initialized clusters.
     */
    virtual void maximize() = 0;
};

template <typename precision, typename int_size>
void AbstractKmeansAlgorithm<precision, int_size>::setKmeansData(KmeansData<precision, int_size>* kmeansData)
{
    pKmeansData = kmeansData;
    // pData            = kmeansData->p_Data;
    // pWeights         = kmeansData->p_Weights;
    // ppClusters       = &kmeansData->p_Clusters;
    // ppClustering     = &kmeansData->p_Clustering;
    // ppClusterWeights = &kmeansData->p_ClusterWeights;
    // ppSqDistances    = &kmeansData->p_SqDistances;
    // pLengths         = &kmeansData->m_Lengths;
    // pRank            = &kmeansData->m_Rank;
    // pTotalNumData    = &kmeansData->m_TotalNumData;
    // pDisplacements   = &kmeansData->displacements;
    // pDistanceFunc    = kmeansData->distanceFunc;
}
}  // namespace HPKmeans