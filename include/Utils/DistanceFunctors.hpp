#pragma once

#include <math.h>

#include "Containers/Definitions.hpp"

/**
 * @brief An interface for functor classes that calculate distances between points.
 */
class IDistanceFunctor
{
public:
    /**
     * @brief Overloaded function call operator.
     *
     * @param point1 - The first datapoint.
     * @param point2 - The second datapoint.
     * @param numFeatures - The number of features in each datapoint.
     * @return value_t - The distance between each point.
     */
    virtual value_t operator()(const value_t* point1, const value_t* point2, const int32_t& numFeatures) = 0;
};

/**
 * @brief Implementation of IDistanceFunctor that calculates the Euclidean distance between two points.
 */
class EuclideanDistance : public IDistanceFunctor
{
public:
    /**
     * @brief Overloaded function call operator that calculates the Euclidean distance between two points.
     *
     * @param point1 - The first datapoint.
     * @param point2 - The second datapoint.
     * @param numFeatures - The number of features in each datapoint.
     * @return value_t
     */
    value_t operator()(const value_t* point1, const value_t* point2, const int32_t& numFeatures)
    {
        value_t sum = 0;
        for (int32_t i = 0; i < numFeatures; i++)
        {
            sum += std::pow(point1[i] - point2[i], 2);
        }

        return std::sqrt(sum);
    }
};