#pragma once

#include "AbstractKmeansAlgorithms.hpp"

/**
 * @brief Implementation of a Kmeans initialization aglorithm. Selects datapoints to be new clusters at random weighted
 *        by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 *
 */
class SerialKPlusPlus : public AbstractKmeansInitializer
{
public:
    /**
     * @brief Top level function that initializes the clusters.
     *
     * @param distanceFunc - The functor that defines the distance metric to use.
     * @param seed - The seed for the RNG.
     */
    void initialize(IDistanceFunctor *distanceFunc, const float &seed);
};