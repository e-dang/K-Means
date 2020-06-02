#pragma once

#include <kmeans/utils/assignment_updaters/abstract_updater.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class AssignmentUpdater : public AbstractAssignmentUpdater<T, DistanceFunc>
{
public:
    AssignmentUpdater() : AbstractAssignmentUpdater<T, DistanceFunc>() {}

    template <Parallelism _Level>
    std::enable_if_t<_Level == Parallelism::Serial> update(const Matrix<T>* const data,
                                                           const Matrix<T>* const centroids,
                                                           std::vector<int32_t>* const assignments,
                                                           std::vector<T>* const sqDistances) const
    {
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            this->updateClosestCentroid(i, centroids, assignments, sqDistances);
        }
    }

    template <Parallelism _Level>
    std::enable_if_t<_Level == Parallelism::OMP> update(const Matrix<T>* const data, const Matrix<T>* const centroids,
                                                        std::vector<int32_t>* const assignments,
                                                        std::vector<T>* const sqDistances) const
    {
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            this->updateClosestCentroid(i, centroids, assignments, sqDistances);
        }
    }
};
}  // namespace hpkmeans