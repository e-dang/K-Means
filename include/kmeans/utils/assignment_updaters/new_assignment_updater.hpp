#pragma once

#include <kmeans/utils/assignment_updaters/abstract_updater.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class NewCentroidAssignmentUpdater : public AbstractAssignmentUpdater<T, DistanceFunc>
{
public:
    NewCentroidAssignmentUpdater() : AbstractAssignmentUpdater<T, DistanceFunc>() {}

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
        auto newCentroidIdx = centroids->numRows() - 1;
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            updateIteration(i, newCentroidIdx, data, centroids, assignments, sqDistances);
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<_Level == Parallelism::OMP> updateImpl(const Matrix<T>* const data,
                                                            const Matrix<T>* const centroids,
                                                            std::vector<int32_t>* const assignments,
                                                            std::vector<T>* const sqDistances) const
    {
        auto newCentroidIdx = centroids->numRows() - 1;

#pragma omp parallel for shared(newCentroidIdx), schedule(static)
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            updateIteration(i, newCentroidIdx, data, centroids, assignments, sqDistances);
        }
    }

    void updateIteration(const int32_t dataIdx, const int32_t centroidIdx, const Matrix<T>* const data,
                         const Matrix<T>* const centroids, std::vector<int32_t>* const assignments,
                         std::vector<T>* const sqDistances) const
    {
        auto sqDist = std::pow(this->m_distanceFunc(data->crowBegin(dataIdx), data->crowEnd(dataIdx),
                                                    centroids->crowBegin(centroidIdx), centroids->crowEnd(centroidIdx)),
                               2);
        if (sqDist < sqDistances->at(dataIdx))
        {
            assignments->at(dataIdx) = centroidIdx;
            sqDistances->at(dataIdx) = sqDist;
        }
    }
};
}  // namespace hpkmeans