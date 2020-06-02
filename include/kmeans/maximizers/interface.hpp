#pragma once

namespace hpkmeans
{
template <typename T>
class IMaximizer
{
public:
    virtual ~IMaximizer() = default;

    virtual void maximize(const Matrix<T>* const data, Clusters<T>* const clusters) const = 0;

    const T MIN_PERCENT_CHANGED = 0.0001;
};
}  // namespace hpkmeans