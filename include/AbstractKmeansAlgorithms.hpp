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
    Matrix *matrix;
    Matrix *clusters;
    std::vector<int> *clustering;
    std::vector<value_t> *clusterWeights;
    std::vector<value_t> *weights;

public:
    /**
     * @brief Function that calls protected member functions setMatrix(), setClusterData(), and setWeights() with
     *        the given arguments, in order to initialize protected member variables.
     *
     * @param matrix - The data to be clustered.
     * @param clusterData - The struct where the clustering data is going to be stored for a given run.
     * @param weights - The weights for individual datapoints.
     */
    void setUp(Matrix *matrix, ClusterData *clusterData, std::vector<value_t> *weights)
    {
        setMatrix(matrix);
        setClusterData(clusterData);
        setWeights(weights);
    }

protected:
    /**
     * @brief Set the matrix member variable.
     *
     * @param matrix - The data to be clustered.
     */
    void setMatrix(Matrix *matrix)
    {
        this->matrix = matrix;
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

    /**
     * @brief Set the weights member variable.
     *
     * @param weights - A pointer to the vector of weights for each datapoint.
     */
    void setWeights(std::vector<value_t> *weights)
    {
        this->weights = weights;
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

        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0 && clusterWeights->at(clusterAssignment) > 0)
            clusterWeights->at(clusterAssignment) -= weights->at(dataIdx);
        clusterWeights->at(clusterIdx) += weights->at(dataIdx);
        clusterAssignment = clusterIdx;
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
     * @brief Interface that Kmeans initialization algorithms must follow for initializing the clusters.
     *
     * @param distanceFunc - A pointer to a class that calculates distances between points and is an implementation of
     *                       IDistanceFunctor.
     * @param seed - The number to seed the RNG.
     */
    virtual void initialize(IDistanceFunctor *distanceFunc, const float &seed) = 0;
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
     * @brief Interface that Kmeans maximization algorithms must follow for finding the best clustering given a set of
     *        pre-initialized clusters.
     *
     * @param distanceFunc - A pointer to a class that calculates distances between points and is an implementation of
     *                       IDistanceFunctor.
     * @return std::vector<value_t> - A vector containing the square distances of each point to its nearest cluster.
     */
    virtual std::vector<value_t> maximize(IDistanceFunctor *distanceFunc) = 0;
};