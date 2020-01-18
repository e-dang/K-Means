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
     *
     */
    ~Kmeans(){};

    /**
     * @brief Top level function that initiates the clustering of the passed in data.
     *
     * @param matrix - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of repeats to perform.
     * @param weights - The weights corresponding to each datapoint. Defaults to NULL.
     */
    void fit(Matrix *matrix, int numClusters, int numRestarts, std::vector<value_t> *weights = NULL);
};