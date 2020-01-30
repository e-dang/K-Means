#pragma once

#include "Definitions.hpp"
#include "DataClasses.hpp"
#include "AbstractKmeansAlgorithms.hpp"
#include "DistanceFunctors.hpp"
#include <numeric>

/**
 * @brief Abstract class that defines the interface for using a Kmeans class, which wraps an initialization and
 *        maximization algorithm together, along with a distance metric functor in order to cluster data. In addition
 *        this class also defines the member variables, setters, getters, and helper functions that each Kmeans
 *        concretion will need to function.
 */
class AbstractKmeans
{
protected:
    // Member variables
    AbstractKmeansInitializer *initializer;
    AbstractKmeansMaximizer *maximizer;
    IDistanceFunctor *distanceFunc;
    ClusterData finalClusterData;
    double error;

public:
    /**
     * @brief Constructor for AbstractKmeans.
     *
     * @param initializer - A pointer to a class implementing a Kmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a Kmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    AbstractKmeans(AbstractKmeansInitializer *initializer, AbstractKmeansMaximizer *maximizer,
                   IDistanceFunctor *distanceFunc) : initializer(initializer),
                                                     maximizer(maximizer),
                                                     distanceFunc(distanceFunc),
                                                     error(-1) {}

    /**
     * @brief Destroy the Abstract Kmeans object.
     */
    virtual ~AbstractKmeans(){};

    /**
     * @brief Overloaded interface for the top level function that initiates the clustering process, where the weights
     *        of each datapoint is unspecified. This function should form a vector of uniform weights and pass it to
     *        the fit() function that takes weights as parameter, but the specific implementation is left up to the
     *        concretion of this class.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of times to repeat the clustering process.
     */
    virtual void fit(Matrix *matrix, int numClusters, int numRestarts) = 0;

    /**
     * @brief Interface for the top level function that initiates the clustering process.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of times to repeat the clustering process.
     * @param weights - The weights for each datapoint in the matrix.
     */
    virtual void fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights) = 0;

    /**
     * @brief Get the finalClusterData member variable.
     *
     * @return ClusterData
     */
    ClusterData getClusterData() { return finalClusterData; }

    /**
     * @brief Get the error of the resulting clustering.
     *
     * @return double
     */
    double getError() { return error; }

    /**
     * @brief Set the initializer member variable.
     *
     * @param initializer - A pointer to an implementation of the AbstractKmeansInitializer class.
     */
    void setInitializer(AbstractKmeansInitializer *initializer) { this->initializer = initializer; }

    /**
     * @brief Set the maximizer member variable.
     *
     * @param maximizer - A pointer to an implementation of the AbstractKmeansMaximizer class.
     */
    void setMaximizer(AbstractKmeansMaximizer *maximizer) { this->maximizer = maximizer; }

    /**
     * @brief Set the distanceFunc member variable.
     *
     * @param distanceFunc - A pointer to an implementation of the IDistanceFunctor class.
     */
    void setDistanceFunc(IDistanceFunctor *distanceFunc) { this->distanceFunc = distanceFunc; }

protected:
    /**
     * @brief Helper function that takes in the resulting clusterData and squared distances of each datapoint to their
     *        assigned cluster and calculates the error. If the error is less than the previous run's error, the
     *        clusterData from the current run is stored in finalClusterData.
     *
     * @param clusterData - The clusterData from the current run.
     * @param distances - The square distances from each to point to their assigned cluster.
     */
    void compareResults(ClusterData *clusterData, std::vector<value_t> *distances)
    {
        double currError = std::accumulate(distances->begin(), distances->end(), 0);

        if (error > currError || error < 0)
        {
            error = currError;
            finalClusterData = *clusterData;
        }
    }

    virtual StaticData initStaticData(Matrix *data, std::vector<value_t> *weights)
    {
        return StaticData{data,
                          weights,
                          distanceFunc,
                          0,
                          data->getMaxNumData(),
                          std::vector<int>(1, data->getMaxNumData()),
                          std::vector<int>(1, 0)};
    }
};