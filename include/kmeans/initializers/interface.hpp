#pragma once

#include <kmeans/types/clusters.hpp>
#include <matrix/matrix.hpp>

namespace hpkmeans
{
template <typename T>
class IInitializer
{
public:
    virtual ~IInitializer() = default;

    virtual void initialize(const Matrix<T>* const data, Clusters<T>* const clusters) const = 0;
};
}  // namespace hpkmeans