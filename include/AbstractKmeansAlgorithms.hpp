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
    // /**
    //  * @brief Set the clusters, clustering, and clusterWeights member variables using an instance of ClusterData.
    //  *
    //  * @param clusterData - A pointer to an instance of clusterData, where the clustering results will be stored.
    //  */
    // void setClusterData(ClusterData *clusterData);

    // /**
    //  * @brief Helper function that updates the clustering assignments and cluster weights given the index of the
    //  *        datapoint whose clustering assignment has been changed and the index of the new cluster it is assigned to.
    //  *
    //  * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
    //  * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
    //  */
    // virtual void updateClustering(const int &dataIdx, const int &clusterIdx);

    // /**
    //  * @brief Helper function that returns the current number of clusters stored in the clusters member variable. Since
    //  *        the clusters are stored in a flattened array, the number of clusters is equal to the the size of the array
    //  *        divided by the number of columns of the matrix.
    //  *
    //  * @return int - The current number of clusters.
    //  */
    // int getCurrentNumClusters();

    // virtual value_t calcDistance(const int &dataIdx, const int &clusterIdx, IDistanceFunctor *distanceFunc);

    // /**
    //  * @brief Helper function that find the closest cluster and corresponding distance for a given datapoint.
    //  *
    //  * @param dataIdx - A the index of the datapoint that the function will find the closest cluster to.
    //  * @param distanceFunc - A functor that defines the distance metric.
    //  * @return ClosestCluster - struct containing the cluster index of the closest cluster and the corresponding distance.
    //  */
    // ClosestCluster findClosestCluster(const int &dataIdx, IDistanceFunctor *distanceFunc);
protected:
    void findAndUpdateClosestCluster(const int &dataIdx)
    {
        auto closestCluster = pFinder->findClosestCluster(pData->at(dataIdx), pDistanceFunc);
        pUpdater->update(dataIdx, closestCluster.clusterIdx, pWeights->at(dataIdx));
        pDistances->at(dataIdx) = std::pow(closestCluster.distance, 2);
    }
};

/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
class AbstractKmeansInitializer : public AbstractKmeansAlgorithm
{
public:
    // AbstractKmeansInitializer() {}
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

// /**
//  * @brief An abstract class that should also be inherited along with AbstractKmeansAlgorithm for classes that use OMP
//  *        thread level parallelism. This class offers an atomic version of functions that have race conditions.
//  */
// class AbstractOMPKmeansAlgorithm : public virtual AbstractKmeansAlgorithm
// {
// public:
//     /**
//      * @brief Destroy the AbstractOMPKmeansAlgorithm object.
//      *
//      */
//     ~AbstractOMPKmeansAlgorithm(){};

//     /**
//      * @brief Atomic version of updateClustering for Kmeans algorithm classes who use OMP thread level parallelism.
//      *
//      * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
//      * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
//      */
//     void updateClustering(const int &dataIdx, const int &clusterIdx) override;
// };

// class AbstractMPIKmeansAlgorithm : public virtual AbstractKmeansAlgorithm
// {
// protected:
//     // Member variables
//     int mRank;
//     Matrix *pMatrixChunk;
//     std::vector<int> *pLengths;
//     std::vector<int> *pDisplacements;

// public:
//     /**
//      * @brief Destroy the AbstractMPIKmeansAlgorithm object
//      */
//     virtual ~AbstractMPIKmeansAlgorithm(){};

//     void setUp(BundledAlgorithmData *bundledData) override;

//     value_t calcDistance(const int &dataIdx, const int &clusterIdx, IDistanceFunctor *distanceFunc) override;

//     /**
//      * @brief MPI version of updateClustering for Kmeans algorithm classes who use MPI process level parallelism.
//      *
//      * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
//      * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
//      */
//     void updateClustering(const int &dataIdx, const int &clusterIdx) override;

// protected:
//     void bcastClusterData();
// };

// class AbstractWeightedClusterSelection : public AbstractKmeansAlgorithm
// {
// protected:
//     // Member variables
//     AbstractKmeansInitializer *pAlg;

// public:
//     AbstractWeightedClusterSelection(AbstractKmeansInitializer *alg) : pAlg(alg){};
//     /**
//      * @brief Destroy the AbstractWeightedClusterSelection object
//      *
//      */
//     virtual ~AbstractWeightedClusterSelection(){};

//     virtual void weightedClusterSelection(std::vector<value_t> *distances, float &randSumFrac) = 0;
// };

// class WeightedClusterSelection : public AbstractWeightedClusterSelection
// {
// public:
//     WeightedClusterSelection(AbstractKmeansInitializer *alg) : AbstractWeightedClusterSelection(alg){};
//     ~WeightedClusterSelection(){};

//     void weightedClusterSelection(std::vector<value_t> *distances, float &randSumFrac) override;
// };

// class AbstractFindAndUpdate : public AbstractKmeansAlgorithm
// {
// protected:
//     AbstractKmeansAlgorithm *pAlg;

// public:
//     AbstractFindAndUpdate(AbstractKmeansAlgorithm *alg) : pAlg(alg){};
//     virtual ~AbstractFindAndUpdate(){};

//     virtual void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) = 0;
// };

// class FindAndUpdateClosestCluster : public AbstractFindAndUpdate
// {
// public:
//     FindAndUpdateClosestCluster(AbstractKmeansAlgorithm *alg) : AbstractFindAndUpdate(alg){};
//     ~FindAndUpdateClosestCluster(){};

//     void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;
// };

// class OptFindAndUpdateClosestCluster : public AbstractFindAndUpdate
// {
// public:
//     OptFindAndUpdateClosestCluster(AbstractKmeansAlgorithm *alg) : AbstractFindAndUpdate(alg){};
//     ~OptFindAndUpdateClosestCluster(){};

//     void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
//                                      IDistanceFunctor *distanceFunc) override;
// };

// class ReassignmentClosestClusterFinder : public AbstractFindAndUpdate
// {
// public:
//     ReassignmentClosestClusterFinder(AbstractKmeansAlgorithm *alg) : AbstractFindAndUpdate(alg){};
//     ~ReassignmentClosestClusterFinder(){};

//     void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
//                                      IDistanceFunctor *distanceFunc) override;
// };