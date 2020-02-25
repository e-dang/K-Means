#pragma once

#include <hpkmeans/data_types/kmeans_state.hpp>
#include <memory>

namespace HPKmeans
{
/**
 * @brief Abstract class that all Kmeans algorithms, such as initializers and maximizers will derive from. This class
 *        contains code that is used to set up each of these algorithms.
 */
template <typename precision, typename int_size>
class AbstractKmeansAlgorithm
{
protected:
    KmeansState<precision, int_size>* p_KmeansState;

public:
    AbstractKmeansAlgorithm() = default;

    virtual ~AbstractKmeansAlgorithm() = default;

    void setState(KmeansState<precision, int_size>* kmeansState);
};

template <typename precision, typename int_size>
void AbstractKmeansAlgorithm<precision, int_size>::setState(KmeansState<precision, int_size>* kmeansState)
{
    p_KmeansState = kmeansState;
}
}  // namespace HPKmeans