#pragma once

#include <kmeans/coreset/coreset_creator_impl.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class SharedMemoryCoresetCreatorImpl : public AbstractCoresetCreatorImpl<T, Level, DistanceFunc>
{
public:
    SharedMemoryCoresetCreatorImpl() : AbstractCoresetCreatorImpl<T, Level, DistanceFunc>() {}

    Coreset<T> createCoreset(const Matrix<T>* const data, const int32_t& sampleSize) override
    {
        auto mean = calcMean(data);
        std::vector<T> sqDistances(data->numRows());
        auto distanceSum  = this->calcDistsFromMean(data, mean, sqDistances);
        auto distribution = calcDistribution(sqDistances, distanceSum);

        return sampleDistribution(data, distribution, sampleSize);
    }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level), std::vector<T>> calcMean(const Matrix<T>* const data)
    {
        std::vector<T> mean(data->cols(), 0.0);

        for (int32_t i = 0; i < data->cols(); ++i)
        {
            mean[i] = std::accumulate(data->ccolBegin(i), data->ccolEnd(i), 0.0) / static_cast<T>(data->numRows());
        }

        return mean;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level), std::vector<T>> calcMean(const Matrix<T>* const data)
    {
        std::vector<T> mean(data->cols(), 0.0);

#pragma omp parallel for schedule(static), shared(mean)
        for (int32_t i = 0; i < data->cols(); ++i)
        {
            mean[i] = std::accumulate(data->ccolBegin(i), data->ccolEnd(i), 0.0) / static_cast<T>(data->numRows());
        }

        return mean;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level), std::vector<T>> calcDistribution(const std::vector<T>& sqDistances,
                                                                                const T& distanceSum)
    {
        std::vector<T> distribution(sqDistances.size());
        T partialQ = 0.5 * (1.0 / static_cast<T>(sqDistances.size()));
        std::transform(sqDistances.cbegin(), sqDistances.cend(), distribution.begin(),
                       [&partialQ, &distanceSum](const T& dist) { return partialQ + (0.5 * dist / distanceSum); });

        return distribution;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level), std::vector<T>> calcDistribution(const std::vector<T>& sqDistances,
                                                                               const T& distanceSum)
    {
        std::vector<T> distribution(sqDistances.size());
        T partialQ = 0.5 * (1.0 / static_cast<T>(sqDistances.size()));

#pragma omp parallel for schedule(static), shared(partialQ, distribution)
        for (int32_t i = 0; i < static_cast<int32_t>(sqDistances.size()); ++i)
        {
            distribution[i] = partialQ + (0.5 * sqDistances[i] / distanceSum);
        }

        return distribution;
    }

    Coreset<T> sampleDistribution(const Matrix<T>* const data, const std::vector<T>& distribution,
                                  const int32_t& sampleSize)
    {
        Coreset<T> coreset(sampleSize, data->cols());

        auto selectedIdxs = this->m_weightedSelector.selectMultiple(&distribution, sampleSize);
        for (const auto& idx : selectedIdxs)
        {
            coreset.append(data->crowBegin(idx), data->crowEnd(idx), 1.0 / (sampleSize * distribution[idx]));
        }

        return coreset;
    }
};

}  // namespace hpkmeans