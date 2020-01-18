#pragma once

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"

class AbstractKmeansAlgorithm
{
protected:
    // member variables
    Matrix *matrix;
    Matrix *clusters;
    std::vector<int> *clustering;
    std::vector<int> *clusterCounts;

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
     * @brief Set the clusters, clustering, and clusterCounts member variables using an instance of ClusterData.
     *
     * @param clusterData - A pointer to an instance of clusterData, where the clustering results will be stored.
     */
    void setClusterData(ClusterData *clusterData)
    {
        clustering = &clusterData->clustering;
        clusters = &clusterData->clusters;
        clusterCounts = &clusterData->clusterCounts;
    }

    /**
     * @brief Helper function that updates the clustering assignments and cluster counts given the index of the
     *        datapoint whose clustering assignment has been changed and the index of the new cluster it is assigned to.
     *
     * @param dataIdx - The index of the datapoint whose clustering assignment needs to be updated.
     * @param clusterIdx - The index of the cluster to which the datapoint is now assigned.
     */
    inline void updateClustering(const int &dataIdx, const int &clusterIdx)
    {
        int &clusterAssignment = clustering->at(dataIdx);

        if (clusterAssignment >= 0 && clusterCounts->at(clusterAssignment) > 0)
            clusterCounts->at(clusterAssignment)--;
        clusterCounts->at(clusterIdx)++;
        clusterAssignment = clusterIdx;
    }
};

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

    /**
     * @brief Helper function that calls protected member functions setMatrix() and setClusterData() with the given
     *        arguments. This should be called before intialize() is called.
     *
     * @param matrix - The data to be clustered.
     * @param clusterData - The struct where the clustering data is going to be stored for a given run.
     */
    void setUp(Matrix *matrix, ClusterData *clusterData)
    {
        setMatrix(matrix);
        setClusterData(clusterData);
    }
};

class AbstractKmeansMaximizer : public AbstractKmeansAlgorithm
{
protected:
    // member variables
    std::vector<value_t> *weights;

public:
    /**
     * @brief Interface that Kmeans maximization algorithms must follow for finding the best clustering given a set of
     *        pre-initialized clusters.
     *
     * @param distanceFunc - A pointer to a class that calculates distances between points and is an implementation of
     *                       IDistanceFunctor.
     * @return std::vector<value_t>
     */
    virtual std::vector<value_t> maximize(IDistanceFunctor *distanceFunc) = 0;

    /**
     * @brief Helper function that calls protected member functions setMatrix(), setClusterData(), and setWeights() with
     *         the given arguments. This should be called before intialize() is called.
     *
     * @param matrix - The data to be clustered.
     * @param clusterData - The struct where the clustering data is going to be stored for a given run.
     * @param weights - The weights for individual datapoints.
     */
    void setUp(Matrix *matrix, ClusterData *clusterData, std::vector<value_t> *weights)
    {
        setMatrix(matrix);
        setClusterData(clusterData);
        setWeights(weights, matrix->numRows);
    }

protected:
    /**
     * @brief Set the weights member variable. If the argument weights is NULL, then a vector of length numData is
     *        created with all values equal to 1 and this vector is used as the weights for the data.
     *
     * @param weights - A pointer to the vector of weights for each datapoint.
     * @param numData - The number of datapoints that are going to be clustered.
     */
    void setWeights(std::vector<value_t> *weights, const int &numData)
    {
        if (weights == NULL)
        {
            std::vector<value_t> vectorWeights(numData, 1);
            weights = &vectorWeights;
        }

        this->weights = weights;
    }
};