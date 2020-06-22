#pragma once

#include <hpkmeans/types/clusters.hpp>
#include <matrix/matrix.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level>
class IInitializer
{
public:
    virtual ~IInitializer() = default;

    virtual void initialize(const Matrix<T>* const data, Clusters<T, Level>* const clusters) const = 0;
};
}  // namespace hpkmeans