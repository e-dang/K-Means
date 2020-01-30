#pragma once

#include "AbstractKmeans.hpp"

/**
 * @brief Implementation of AbstractKmeans that can take serialized or threaded implementations of KmeansAlgorithms.
 */
class Kmeans : public AbstractKmeans
{
public:
    /**
     * @brief Construct a new Kmeans object. Calls base class' constructor.
     *
     * @param initializer - A pointer to a class implementing a Kmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a Kmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    Kmeans(AbstractKmeansInitializer *initializer, AbstractKmeansMaximizer *maximizer,
           IDistanceFunctor *distanceFunc) : AbstractKmeans(initializer, maximizer, distanceFunc) {}

    /**
     * @brief Destroy the Kmeans object.
     */
    virtual ~Kmeans(){};

    /**
     * @brief Overloaded top level function that initiates the clustering of the passed in data if weights are not
     *        specified. Thus this function creates a set of default weights (all equal to 1) and calls the overloaded
     *        fit() variant that takes weights.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of repeats to perform.
     */
    void fit(Matrix *matrix, int numClusters, int numRestarts) override;

    /**
     * @brief Overloaded top level function that initiates the clustering of the passed in data if weights are
     *        specified.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of repeats to perform.
     * @param weights - The weights corresponding to each datapoint. Defaults to NULL.
     */
    void fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights) override;
};

class MPIKmeans : public AbstractKmeans
{
private:
    int mTotalNumData;

public:
    /**
     * @brief Construct a new MPIKmeans object. Calls base class' constructor.
     *
     * @param initializer - A pointer to a class implementing a MPIKmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a MPIKmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    MPIKmeans(const int &totalNumData, AbstractKmeansInitializer *initializer, AbstractKmeansMaximizer *maximizer,
              IDistanceFunctor *distanceFunc) : mTotalNumData(totalNumData),
                                                AbstractKmeans(initializer, maximizer, distanceFunc) {}

    /**
     * @brief Destroy the MPIKmeans object.
     */
    virtual ~MPIKmeans(){};

    /**
     * @brief Overloaded top level function that initiates the clustering of the passed in data if weights are not
     *        specified. Thus this function creates a set of default weights (all equal to 1) and calls the overloaded
     *        fit() variant that takes weights.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of repeats to perform.
     */
    void fit(Matrix *matrix, int numClusters, int numRestarts) override;

    /**
     * @brief Overloaded top level function that initiates the clustering of the passed in data if weights are
     *        specified.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of repeats to perform.
     * @param weights - The weights corresponding to each datapoint. Defaults to NULL.
     */
    void fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights) override;

protected:
    StaticData initStaticData(Matrix *data, std::vector<value_t> *weights) override;
};