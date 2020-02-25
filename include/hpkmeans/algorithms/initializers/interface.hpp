#pragma once

#include <hpkmeans/algorithms/kmeans_algorithm.hpp>
#include <hpkmeans/algorithms/strategies/ClosestClusterUpdater.hpp>

namespace HPKmeans
{
/**
 * @brief Abstract class that defines the interface for Kmeans initialization algorithms, such as K++ or random
 *        initialization.
 */
template <typename precision, typename int_size>
class IKmeansInitializer : public AbstractKmeansAlgorithm<precision, int_size>
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> pUpdater;

public:
    IKmeansInitializer(AbstractClosestClusterUpdater<precision, int_size>* updater) : pUpdater(updater) {}

    virtual ~IKmeansInitializer() = default;

    /**
     * @brief Interface that Kmeans initialization algorithms must follow for initializing the clusters.
     */
    virtual void initialize() = 0;
};
}  // namespace HPKmeans