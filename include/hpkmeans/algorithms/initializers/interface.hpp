#pragma once

#include <hpkmeans/algorithms/kmeans_algorithm.hpp>

namespace HPKmeans
{
/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
template <typename precision, typename int_size>
class IKmeansInitializer : public AbstractKmeansAlgorithm<precision, int_size>
{
public:
    virtual ~IKmeansInitializer() = default;

    /**
     * @brief Interface that Kmeans initialization algorithms must follow for initializing the clusters.
     */
    virtual void initialize() = 0;
};
}  // namespace HPKmeans