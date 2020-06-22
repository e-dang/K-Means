#pragma once

#include <hpkmeans/utils/assignment_updaters/abstract_updater.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class AssignmentUpdater : public AbstractAssignmentUpdater<T, DistanceFunc>
{
public:
    AssignmentUpdater() : AbstractAssignmentUpdater<T, DistanceFunc>() {}

    void update(const Matrix<T>* const data, const Matrix<T>* const centroids, VectorView<int32_t>* const assignments,
                VectorView<T>* const sqDistances) const override
    {
        updateImpl(data, centroids, assignments, sqDistances);
    }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level)> updateImpl(const Matrix<T>* const data, const Matrix<T>* const centroids,
                                                          VectorView<int32_t>* const assignments,
                                                          VectorView<T>* const sqDistances) const
    {
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            this->updateClosestCentroid(i, data, centroids, assignments, sqDistances);
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level)> updateImpl(const Matrix<T>* const data, const Matrix<T>* const centroids,
                                                         VectorView<int32_t>* const assignments,
                                                         VectorView<T>* const sqDistances) const
    {
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            this->updateClosestCentroid(i, data, centroids, assignments, sqDistances);
        }
    }
};
}  // namespace hpkmeans