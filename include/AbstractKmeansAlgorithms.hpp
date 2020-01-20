#pragma once

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"
#include "mpi.h"

/**
 * @brief Abstract class that all Kmeans algorithms, such as initializers and maximizers will derive from. This class
 *        contains code that is used to set up each of these algorithms.
 */
class AbstractKmeansAlgorithm
{
protected:
    // Member variables
    Matrix *matrix;
    Matrix *clusters;
    std::vector<int> *clustering;
    std::vector<value_t> *clusterWeights;
    std::vector<value_t> *weights;

    friend class AbstractOMPKmeansAlgorithm;
    friend class AbstractMPIKmeansAlgorithm;

public:
    /**
     * @brief Destroy the AbstractKmeansAlgorithm object
     */
    virtual ~AbstractKmeansAlgorithm(){};

    // /**
    //  * @brief Function that calls protected member functions setMatrix(), setClusterData(), and setWeights() with
    //  *        the given arguments, in order to initialize protected member variables.
    //  *
    //  * @param matrix - The data to be clustered.
    //  * @param clusterData - The struct where the clustering data is going to be stored for a given run.
    //  * @param weights - The weights for individual datapoints.
    //  */
    // virtual void setUp(Matrix *matrix, ClusterData *clusterData, std::vector<value_t> *weights)
    // {
    //     setMatrix(matrix);
    //     setClusterData(clusterData);
    //     setWeights(weights);
    // }

    /**
     * @brief Function that calls protected member functions setMatrix(), setClusterData(), and setWeights() with
     *        the given arguments, in order to initialize protected member variables.
     *
     * @param matrix - The data to be clustered.
     * @param clusterData - The struct where the clustering data is going to be stored for a given run.
     * @param weights - The weights for individual datapoints.
     */
    virtual void setUp(BundledAlgorithmData *bundledData)
    {
        this->matrix = bundledData->matrix;
        this->weights = bundledData->weights;
    }

    /**
     * @brief Set the clusters, clustering, and clusterWeights member variables using an instance of ClusterData.
     *
     * @param clusterData - A pointer to an instance of clusterData, where the clustering results will be stored.
     */
    void setClusterData(ClusterData *clusterData)
    {
        clustering = &clusterData->clustering;
        clusters = &clusterData->clusters;
        clusterWeights = &clusterData->clusterWeights;
    }

protected:
    /**
     * @brief Helper function that returns the current number of clusters stored in the clusters member variable. Since
     *        the clusters are stored in a flattened array, the number of clusters is equal to the the size of the array
     *        divided by the number of columns of the matrix.
     *
     * @return int - The current number of clusters.
     */
    int getCurrentNumClusters() { return clusters->data.size() / clusters->numCols; }

    /**
     * @brief Helper function that find the closest cluster and corresponding distance for a given datapoint.
     *
     * @param dataIdx - A the index of the datapoint that the function will find the closest cluster to.
     * @param distanceFunc - A functor that defines the distance metric.
     * @return ClosestCluster - struct containing the cluster index of the closest cluster and the corresponding distance.
     */
    ClosestCluster findClosestCluster(const int &dataIdx, IDistanceFunctor *distanceFunc)
    {
        int clusterIdx, numExistingClusters = getCurrentNumClusters();
        value_t tempDistance, minDistance = -1;

        for (int i = 0; i < numExistingClusters; i++)
        {
            tempDistance = (*distanceFunc)(&*matrix->at(dataIdx), &*clusters->at(i), clusters->numCols);

            if (minDistance > tempDistance || minDistance < 0)
            {
                minDistance = tempDistance;
                clusterIdx = i;
            }
        }

        return ClosestCluster{clusterIdx, minDistance};
    }

    /**
     * @brief Helper function that find the closest cluster and corresponding distance for a given datapoint.
     *
     * @param dataIdx - A the index of the datapoint that the function will find the closest cluster to.
     * @param distanceFunc - A functor that defines the distance metric.
     * @return ClosestCluster - struct containing the cluster index of the closest cluster and the corresponding distance.
     */
    ClosestCluster findClosestCluster(value_t *datapoint, IDistanceFunctor *distanceFunc)
    {
        int clusterIdx, numExistingClusters = getCurrentNumClusters();
        value_t tempDistance, minDistance = -1;

        for (int i = 0; i < numExistingClusters; i++)
        {
            tempDistance = (*distanceFunc)(datapoint, &*clusters->at(i), clusters->numCols);

            if (minDistance > tempDistance || minDistance < 0)
            {
                minDistance = tempDistance;
                clusterIdx = i;
            }
        }

        return ClosestCluster{clusterIdx, minDistance};
    }

    /**
     * @brief Helper function that updates the clustering assignments and cluster weights given the index of the
     *        datapoint whose clustering assignment has been changed and the index of the new cluster it is assigned to.
     *
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    inline void updateClustering(const int &dataIdx, const int &clusterIdx)
    {
        int &clusterAssignment = clustering->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0 && clusterWeights->at(clusterAssignment) > 0)
                clusterWeights->at(clusterAssignment) -= weights->at(dataIdx);
            clusterWeights->at(clusterIdx) += weights->at(dataIdx);
            clusterAssignment = clusterIdx;
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
};

/**
 * @brief An abstract class that should also be inherited along with AbstractKmeansAlgorithm for classes that use OMP
 *        thread level parallelism. This class offers an atomic version of functions that have race conditions.
 */
class AbstractOMPKmeansAlgorithm
{
protected:
    /**
     * @brief Atomic version of updateClustering for Kmeans algorithm classes who use OMP thread level parallelism.
     *
     * @param alg - An pointer to an instance of AbstractKmeansAlgorithm that uses OMP thread level parallelism.
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    void atomicUpdateClustering(AbstractKmeansAlgorithm *alg, const int &dataIdx, const int &clusterIdx)
    {
        int &clusterAssignment = alg->clustering->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0 && alg->clusterWeights->at(clusterAssignment) > 0)
#pragma omp atomic
                alg->clusterWeights->at(clusterAssignment) -= alg->weights->at(dataIdx);
#pragma omp atomic
            alg->clusterWeights->at(clusterIdx) += alg->weights->at(dataIdx);
            clusterAssignment = clusterIdx;
        }
    }

public:
    /**
     * @brief Destroy the AbstractOMPKmeansAlgorithm object.
     *
     */
    ~AbstractOMPKmeansAlgorithm(){};
};

class AbstractMPIKmeansAlgorithm
{
protected:
    // Member variables
    int rank;
    Matrix *matrixChunk;
    std::vector<int> *lengths;
    std::vector<int> *displacements;

    /**
     * @brief MPI version of updateClustering for Kmeans algorithm classes who use MPI process level parallelism.
     *
     * @param alg - An pointer to an instance of AbstractKmeansAlgorithm that uses MPI process level parallelism.
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    void updateClustering(AbstractKmeansAlgorithm *alg, const int &dataIdx, const int &clusterIdx)
    {
        int &clusterAssignment = alg->clustering->at(dataIdx);

        // only go through this update if the cluster assignment is going to change
        if (clusterAssignment != clusterIdx)
        {
            // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
            if (clusterAssignment >= 0)
                alg->clusterWeights->at(clusterAssignment) -= alg->weights->at(dataIdx);
            alg->clusterWeights->at(clusterIdx) += alg->weights->at(dataIdx);
            clusterAssignment = clusterIdx;
        }
    }

    void setUp(AbstractKmeansAlgorithm *alg, BundledMPIAlgorithmData *bundledData)
    {
        alg->matrix = bundledData->matrix;
        alg->weights = bundledData->weights;
        this->rank = bundledData->dataChunks->rank;
        this->matrixChunk = &bundledData->dataChunks->matrixChunk;
        this->lengths = &bundledData->dataChunks->lengths;
        this->displacements = &bundledData->dataChunks->displacements;
    }

    void bcastClusterData(AbstractKmeansAlgorithm *alg)
    {
        MPI_Bcast(alg->clustering->data(), alg->clustering->size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(alg->clusters->data.data(), alg->clusters->data.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
};