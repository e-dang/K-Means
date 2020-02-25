#pragma once

#include <cmath>

namespace HPKmeans
{
/**
 * @brief An interface for functor classes that calculate distances between points.
 */
template <typename precision>
class IDistanceFunctor
{
public:
    virtual ~IDistanceFunctor() = default;

    /**
     * @brief Overloaded function call operator.
     *
     * @param point1 - The first datapoint.
     * @param point2 - The second datapoint.
     * @param numFeatures - The number of features in each datapoint.
     * @return precision - The distance between each point.
     */
    virtual precision operator()(const precision* point1, const precision* point2, const int32_t& numFeatures) = 0;
};

/**
 * @brief Implementation of IDistanceFunctor that calculates the Euclidean distance between two points.
 */
template <typename precision>
class EuclideanDistance : public IDistanceFunctor<precision>
{
public:
    ~EuclideanDistance() = default;

    /**
     * @brief Overloaded function call operator that calculates the Euclidean distance between two points.
     *
     * @param point1 - The first datapoint.
     * @param point2 - The second datapoint.
     * @param numFeatures - The number of features in each datapoint.
     * @return precision
     */
    precision operator()(const precision* point1, const precision* point2, const int32_t& numFeatures)
    {
        precision sum = 0.0;
        for (int32_t i = 0; i < numFeatures; ++i)
        {
            sum += std::pow(point1[i] - point2[i], 2);
        }

        return std::sqrt(sum);
    }
};
}  // namespace HPKmeans