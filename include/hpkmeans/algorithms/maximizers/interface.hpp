#pragma once

#include <hpkmeans/algorithms/kmeans_algorithm.hpp>

namespace HPKmeans
{
/**
 * @brief Abstract class that defines the interface for Kmeans maximization algorithms, such as Lloyd's algorithm.
 */
template <typename precision, typename int_size>
class IKmeansMaximizer : public AbstractKmeansAlgorithm<precision, int_size>
{
protected:
    const precision MIN_PERCENT_CHANGED = 0.0001;  // the % amount of data points allowed to changed before going to
                                                   // next iteration

public:
    virtual ~IKmeansMaximizer() = default;

    /**
     * @brief Interface that Kmeans maximization algorithms must follow for finding the best clustering given a set of
     *        pre-initialized clusters.
     */
    virtual void maximize() = 0;
};
}  // namespace HPKmeans