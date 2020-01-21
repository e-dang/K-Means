#pragma once

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Abstract class that all Kmeans algorithms, such as initializers and maximizers will derive from. This class
 *        contains code that is used to set up each of these algorithms.
 */
class AbstractKmeansAlgorithm
{
protected:
    // Member variables
    Matrix *pMatrix;
    Matrix *pClusters;
    std::vector<int> *pClustering;
    std::vector<value_t> *pClusterWeights;
    std::vector<value_t> *pWeights;

public:
    /**
     * @brief Destroy the AbstractKmeansAlgorithm object
     */
    virtual ~AbstractKmeansAlgorithm(){};

    /**
     * @brief Function that calls protected member functions setMatrix(), setClusterData(), and setWeights() with
     *        the given arguments, in order to initialize protected member variables.
     *
     * @param matrix - The data to be clustered.
     * @param clusterData - The struct where the clustering data is going to be stored for a given run.
     * @param weights - The weights for individual datapoints.
     */
    virtual void setUp(BundledAlgorithmData *bundledData);

    /**
     * @brief Set the clusters, clustering, and clusterWeights member variables using an instance of ClusterData.
     *
     * @param clusterData - A pointer to an instance of clusterData, where the clustering results will be stored.
     */
    void setClusterData(ClusterData *clusterData);

    /**
     * @brief Helper function that updates the clustering assignments and cluster weights given the index of the
     *        datapoint whose clustering assignment has been changed and the index of the new cluster it is assigned to.
     *
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    virtual void updateClustering(const int &dataIdx, const int &clusterIdx);

    /**
     * @brief Helper function that returns the current number of clusters stored in the clusters member variable. Since
     *        the clusters are stored in a flattened array, the number of clusters is equal to the the size of the array
     *        divided by the number of columns of the matrix.
     *
     * @return int - The current number of clusters.
     */
    int getCurrentNumClusters();

    virtual value_t calcDistance(const int &dataIdx, const int &clusterIdx, IDistanceFunctor *distanceFunc);

    /**
     * @brief Helper function that find the closest cluster and corresponding distance for a given datapoint.
     *
     * @param dataIdx - A the index of the datapoint that the function will find the closest cluster to.
     * @param distanceFunc - A functor that defines the distance metric.
     * @return ClosestCluster - struct containing the cluster index of the closest cluster and the corresponding distance.
     */
    ClosestCluster findClosestCluster(const int &dataIdx, IDistanceFunctor *distanceFunc);
};

/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
class AbstractKmeansInitializer : public virtual AbstractKmeansAlgorithm
{
public:
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
    virtual void initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed) = 0;

    void appendCluster(const int &dataIdx);
};

/**
 * @brief Abstract class that defines the interface for Kmeans maximization algorithms, such as Lloyd's algorithm.
 */
class AbstractKmeansMaximizer : public virtual AbstractKmeansAlgorithm
{
protected:
    // Constants
    const float MIN_PERCENT_CHANGED = 0.0001; // the % amount of data points allowed to changed before going to next
                                              // iteration
public:
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
    virtual void maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) = 0;

    void addPointToCluster(const int &dataIdx);

    void averageCluster(const int &clusterIdx);
};

/**
 * @brief An abstract class that should also be inherited along with AbstractKmeansAlgorithm for classes that use OMP
 *        thread level parallelism. This class offers an atomic version of functions that have race conditions.
 */
class AbstractOMPKmeansAlgorithm : public virtual AbstractKmeansAlgorithm
{
public:
    /**
     * @brief Destroy the AbstractOMPKmeansAlgorithm object.
     *
     */
    ~AbstractOMPKmeansAlgorithm(){};

    /**
     * @brief Atomic version of updateClustering for Kmeans algorithm classes who use OMP thread level parallelism.
     *
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    void updateClustering(const int &dataIdx, const int &clusterIdx) override;
};

class AbstractMPIKmeansAlgorithm : public virtual AbstractKmeansAlgorithm
{
protected:
    // Member variables
    int mRank;
    Matrix *pMatrixChunk;
    std::vector<int> *pLengths;
    std::vector<int> *pDisplacements;

public:
    /**
     * @brief Destroy the AbstractMPIKmeansAlgorithm object
     */
    virtual ~AbstractMPIKmeansAlgorithm(){};

    void setUp(BundledAlgorithmData *bundledData) override;

    value_t calcDistance(const int &dataIdx, const int &clusterIdx, IDistanceFunctor *distanceFunc) override;

    /**
     * @brief MPI version of updateClustering for Kmeans algorithm classes who use MPI process level parallelism.
     *
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    void updateClustering(const int &dataIdx, const int &clusterIdx) override;

protected:
    void bcastClusterData();
};

class AbstractWeightedClusterSelection : public AbstractKmeansAlgorithm
{
protected:
    // Member variables
    AbstractKmeansInitializer *pAlg;

public:
    AbstractWeightedClusterSelection(AbstractKmeansInitializer *alg) : pAlg(alg){};
    /**
     * @brief Destroy the AbstractWeightedClusterSelection object
     *
     */
    virtual ~AbstractWeightedClusterSelection(){};

    virtual void weightedClusterSelection(std::vector<value_t> *distances, float &randSumFrac) = 0;
};

class WeightedClusterSelection : public AbstractWeightedClusterSelection
{
public:
    WeightedClusterSelection(AbstractKmeansInitializer *alg) : AbstractWeightedClusterSelection(alg){};
    ~WeightedClusterSelection(){};

    void weightedClusterSelection(std::vector<value_t> *distances, float &randSumFrac) override;
};

class AbstractFindAndUpdate : public AbstractKmeansAlgorithm
{
protected:
    AbstractKmeansAlgorithm *pAlg;

public:
    AbstractFindAndUpdate(AbstractKmeansAlgorithm *alg) : pAlg(alg){};
    virtual ~AbstractFindAndUpdate(){};

    virtual void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) = 0;
};

class FindAndUpdateClosestCluster : public AbstractFindAndUpdate
{
public:
    FindAndUpdateClosestCluster(AbstractKmeansAlgorithm *alg) : AbstractFindAndUpdate(alg){};
    ~FindAndUpdateClosestCluster(){};

    void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances, IDistanceFunctor *distanceFunc) override;
};

class OptFindAndUpdateClosestCluster : public AbstractFindAndUpdate
{
public:
    OptFindAndUpdateClosestCluster(AbstractKmeansAlgorithm *alg) : AbstractFindAndUpdate(alg){};
    ~OptFindAndUpdateClosestCluster(){};

    void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
                                     IDistanceFunctor *distanceFunc) override;
};

class ReassignmentClosestClusterFinder : public AbstractFindAndUpdate
{
public:
    ReassignmentClosestClusterFinder(AbstractKmeansAlgorithm *alg) : AbstractFindAndUpdate(alg){};
    ~ReassignmentClosestClusterFinder(){};

    void findAndUpdateClosestCluster(const int &dataIdx, std::vector<value_t> *distances,
                                     IDistanceFunctor *distanceFunc) override;
};

// class AbstractReassignmentClosestClusterFinder : public AbstractKmeansAlgorithm
// {
// protected:
//     AbstractKmeansAlgorithm *pAlg;

// public:
//     AbstractReassignmentClosestClusterFinder(AbstractKmeansAlgorithm *alg) : pAlg(alg){};
//     virtual ~AbstractReassignmentClosestClusterFinder(){};

//     virtual int findAndUpdateClosestCluster(const int &dataIdx, const int &clusterIdx, std::vector<value_t> *distances,
//                                             IDistanceFunctor *distanceFunc) = 0;
// };

// class ReassignmentClosestClusterFinder : public AbstractReassignmentClosestClusterFinder
// {
//     ReassignmentClosestClusterFinder(AbstractKmeansAlgorithm *alg) : AbstractReassignmentClosestClusterFinder(alg){};
//     ~ReassignmentClosestClusterFinder(){};

//     int findAndUpdateClosestCluster(const int &dataIdx, const int &clusterIdx, std::vector<value_t> *distances,
//                                     IDistanceFunctor *distanceFunc) override;
// };

// class OptReassignmentClosestClusterFinder : public AbstractReassignmentClosestClusterFinder
// {
//     OptReassignmentClosestClusterFinder(AbstractKmeansAlgorithm *alg) : AbstractReassignmentClosestClusterFinder(alg){};
//     ~OptReassignmentClosestClusterFinder(){};

//     int findAndUpdateClosestCluster(const int &dataIdx, const int &clusterIdx, std::vector<value_t> *distances,
//                                     IDistanceFunctor *distanceFunc) override;
// };
// class AbstractCalculateClusterMeans : public AbstractKmeansAlgorithm
// {
// protected:
//     AbstractKmeansMaximizer *pAlg;

// public:
//     AbstractCalculateClusterMeans(AbstractKmeansMaximizer *alg) : pAlg(alg){};
//     ~AbstractCalculateClusterMeans(){};

//     virtual void calculateSum() = 0;
//     virtual void calculateMean() = 0;
// };

// class CalculateClusterMeans : public AbstractCalculateClusterMeans
// {
// public:
//     CalculateClusterMeans(AbstractKmeansMaximizer *alg) : AbstractCalculateClusterMeans(alg){};
//     ~CalculateClusterMeans(){};

//     void calculateSum();
//     void updateClusters();
// };