#pragma once

#include <kmeans/distances.hpp>
#include <kmeans/types/coreset.hpp>
#include <kmeans/types/parallelism.hpp>
#include <kmeans/utils/weighted_selector.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class CoresetCreator
{
public:
    CoresetCreator() : m_distanceFunc(DistanceFunc::instance()) {}

    Coreset<T> createCoreset(const Matrix<T>* const data, const int32_t& sampleSize)
    {
        auto mean = calcMean(data);

        std::vector<T> sqDistances(data->numRows());
        T distanceSum = calcDistsFromMean(data, mean, sqDistances);

        auto distribution = calcDistribution(sqDistances, distanceSum);

        return sampleDistribution(data, distribution, sampleSize);
    }

private:
    std::vector<T> calcMean(const Matrix<T>* const data)
    {
        std::vector<T> mean(data->cols());

        for (int32_t i = 0; i < data->cols(); ++i)
        {
            std::accumulate(data->ccolBegin(i), data->ccolEnd(i), 0.0);
        }

        return mean;
    }

    T calcDistsFromMean(const Matrix<T>* const data, const std::vector<T>& mean, std::vector<T>& sqDistances)
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

    std::vector<T> calcDistribution(const std::vector<T>& sqDistances, const T& distanceSum)
    {
        std::vector<T> distribution(sqDistances.size());
        T partialQ = 0.5 * (1.0 / sqDistances.size());
        std::transform(sqDistances.cbegin(), sqDistances.cend(), distribution.begin(),
                       [&partialQ, &distanceSum](const T& dist) { return partialQ + (0.5 * dist / distanceSum); });

        return distribution;
    }

    Coreset<T> sampleDistribution(const Matrix<T>* const data, const std::vector<T>& distribution,
                                  const int32_t& sampleSize)
    {
        Coreset<T> coreset(sampleSize, data->cols());

        auto selectedIdxs = m_weightedSelector.selectMultiple(&distribution, sampleSize);
        for (const auto& idx : selectedIdxs)
        {
            coreset.append(data->crowBegin(idx), data->crowEnd(idx), 1.0 / (sampleSize * distribution[idx]));
        }

        return coreset;
    }

private:
    DistanceFunc m_distanceFunc;
    WeightedSelector m_weightedSelector;
};
}  // namespace hpkmeans