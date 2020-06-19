#pragma once

#include <kmeans/types/clusters.hpp>
#include <matrix/matrix.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level>
class IMaximizer
{
public:
    virtual ~IMaximizer() = default;

    virtual void maximize(const Matrix<T>* const data, Clusters<T, Level>* const clusters) const = 0;

    const T MIN_PERCENT_CHANGED = 0.0001;
};
}  // namespace hpkmeans