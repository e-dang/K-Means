#pragma once

#include <kmeans/types/parallelism.hpp>
#include <matrix/matrix.hpp>
#include <type_traits>

namespace hpkmeans
{
template <typename T, class DistanceFunc>
class AbstractAssignmentUpdater
{
public:
    virtual void update(const Matrix<T>* const data, const Matrix<T>* const centroids,
                        std::vector<int32_t>* const assignments, std::vector<T>* const sqDistances) const = 0;

    virtual ~AbstractAssignmentUpdater() = default;

protected:
    AbstractAssignmentUpdater() : m_distanceFunc(DistanceFunc::instance()) {}

    void updateClosestCentroid(const int32_t dataIdx, const Matrix<T>* const data, const Matrix<T>* const centroids,
                               std::vector<int32_t>* const assignments, std::vector<T>* const sqDistances) const
    {
        auto closestCentroid = this->findClosestCentroid(data->crowBegin(dataIdx), data->crowEnd(dataIdx), centroids);
        assignments->at(dataIdx) = closestCentroid.idx;
        sqDistances->at(dataIdx) = std::pow(closestCentroid.distance, 2);
    }

private:
    struct ClosestCentroid
    {
        int32_t idx;
        T distance;

        ClosestCentroid() : idx(-1), distance(std::numeric_limits<T>::max()) {}

        bool isGreaterThan(const T otherDist) const { return distance > otherDist; }

        void set(const int32_t idx, const T distance)
        {
            this->idx      = idx;
            this->distance = distance;
        }
    };

    template <typename Iter>
    ClosestCentroid findClosestCentroid(Iter begin, Iter end, const Matrix<T>* const centroids) const
    {
        ClosestCentroid closestCentroid;

        for (int32_t i = 0; i < centroids->numRows(); ++i)
        {
            auto dist = m_distanceFunc(begin, end, centroids->crowBegin(i), centroids->crowEnd(i));
            if (closestCentroid.isGreaterThan(dist))
                closestCentroid.set(i, dist);
        }

        return closestCentroid;
    }

protected:
    DistanceFunc m_distanceFunc;
};
}  // namespace hpkmeans