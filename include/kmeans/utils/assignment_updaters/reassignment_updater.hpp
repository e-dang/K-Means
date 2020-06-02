#pragma once

#include <kmeans/utils/assignment_updaters/reassignment_updater.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class ReassignmentUpdater : public AbstractAssignmentUpdater<T, DistanceFunc>
{
public:
    ReassignmentUpdater() : AbstractAssignmentUpdater<T, DistanceFunc>() {}

    void update(const Matrix<T>* const data, const Matrix<T>* const centroids, std::vector<int32_t>* const assignments,
                std::vector<T>* const sqDistances) const override
    {
        updateImpl(data, centroids, assignments, sqDistances);
    }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<_Level == Parallelism::Serial> updateImpl(const Matrix<T>* const data,
                                                               const Matrix<T>* const centroids,
                                                               std::vector<int32_t>* const assignments,
                                                               std::vector<T>* const sqDistances) const
    {
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            updateIteration(i, data, centroids, assignments, sqDistances);
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<_Level == Parallelism::OMP> updateImpl(const Matrix<T>* const data,
                                                            const Matrix<T>* const centroids,
                                                            std::vector<int32_t>* const assignments,
                                                            std::vector<T>* const sqDistances) const
    {
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            updateIteration(i, data, centroids, assignments, sqDistances);
        }
    }

    void updateIteration(const int32_t dataIdx, const Matrix<T>* const data, const Matrix<T>* const centroids,
                         std::vector<int32_t>* const assignments, std::vector<T>* const sqDistances) const
    {
        auto centroidIdx = assignments->at(dataIdx);
        auto sqDist      = std::pow(this->m_distanceFunc(data->crowBegin(dataIdx), data->crowEnd(dataIdx),
                                                    centroids->crowBegin(centroidIdx), centroids->crowEnd(centroidIdx)),
                               2);
        if (sqDist > sqDistances->at(dataIdx))
            this->updateClosestCentroid(dataIdx, data, centroids, assignments, sqDistances);
        else
            sqDistances->at(dataIdx) = sqDist;
    }
};
}  // namespace hpkmeans