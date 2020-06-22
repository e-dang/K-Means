#pragma once

#include <kmeans/distances.hpp>
#include <kmeans/types/coreset.hpp>
#include <kmeans/types/parallelism.hpp>
#include <kmeans/utils/utils.hpp>
#include <kmeans/utils/weighted_selector.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class AbstractCoresetCreatorImpl
{
public:
    virtual Coreset<T> createCoreset(const Matrix<T>* const data, const int32_t& sampleSize) = 0;

    virtual ~AbstractCoresetCreatorImpl() = default;

protected:
    AbstractCoresetCreatorImpl() : m_distanceFunc(DistanceFunc::instance()) {}

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level), T> calcDistsFromMean(const Matrix<T>* const data,
                                                                    const std::vector<T>& mean,
                                                                    std::vector<T>& sqDistances)
    {
        T distanceSum = 0.0;

        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            sqDistances[i] =
              std::pow(m_distanceFunc(data->crowBegin(i), data->crowEnd(i), mean.cbegin(), mean.cend()), 2);
            distanceSum += sqDistances[i];
        }

        return distanceSum;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level), T> calcDistsFromMean(const Matrix<T>* const data,
                                                                   const std::vector<T>& mean,
                                                                   std::vector<T>& sqDistances)
    {
        T distanceSum = 0.0;

#pragma omp parallel for schedule(static), shared(sqDistances, mean), reduction(+ : distanceSum)
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            sqDistances[i] =
              std::pow(m_distanceFunc(data->crowBegin(i), data->crowEnd(i), mean.cbegin(), mean.cend()), 2);
            distanceSum += sqDistances[i];
        }

        return distanceSum;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level), std::vector<T>> calcMeanSum(const Matrix<T>* const data)
    {
        std::vector<T> meanSum(data->cols(), 0.0);

        for (int32_t i = 0; i < data->cols(); ++i)
        {
            meanSum[i] = std::accumulate(data->ccolBegin(i), data->ccolEnd(i), 0.0);
        }

        return meanSum;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level), std::vector<T>> calcMeanSum(const Matrix<T>* const data)
    {
        std::vector<T> meanSum(data->cols(), 0.0);

#pragma omp parallel for schedule(static), shared(meanSum)
        for (int32_t i = 0; i < data->cols(); ++i)
        {
            meanSum[i] = std::accumulate(data->ccolBegin(i), data->ccolEnd(i), 0.0);
        }

        return meanSum;
    }

protected:
    WeightedSelector m_weightedSelector;

private:
    DistanceFunc m_distanceFunc;
};
}  // namespace hpkmeans