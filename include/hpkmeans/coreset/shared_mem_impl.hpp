#pragma once

#include <hpkmeans/coreset/coreset_creator_impl.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class SharedMemoryCoresetCreatorImpl : public AbstractCoresetCreatorImpl<T, Level, DistanceFunc>
{
public:
    SharedMemoryCoresetCreatorImpl() :
        AbstractCoresetCreatorImpl<T, Level, DistanceFunc>(), p_data(nullptr), m_sampleSize(-1), m_distribution()
    {
    }

    Coreset<T> createCoreset(const Matrix<T>* const data, const int32_t& sampleSize) override
    {
        if (p_data != data || sampleSize != m_sampleSize)
        {
            setState(data, sampleSize);
            auto mean = calcMean(data);
            std::vector<T> sqDistances(data->numRows());
            auto distanceSum = this->calcDistsFromMean(data, mean, sqDistances);
            calcDistribution(sqDistances, distanceSum);
        }

        return sampleDistribution(data);
    }

private:
    void setState(const Matrix<T>* data, const int32_t& sampleSize)
    {
        p_data         = data;
        m_sampleSize   = sampleSize;
        m_distribution = std::vector<T>(data->numRows(), 0.0);
    }

    std::vector<T> calcMean(const Matrix<T>* const data)
    {
        auto mean = this->calcMeanSum(data);
        std::for_each(mean.begin(), mean.end(), [&data](T val) { val /= static_cast<T>(data->numRows()); });
        return mean;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level)> calcDistribution(const std::vector<T>& sqDistances, const T& distanceSum)
    {
        T partialQ = 0.5 * (1.0 / static_cast<T>(sqDistances.size()));
        std::transform(sqDistances.cbegin(), sqDistances.cend(), m_distribution.begin(),
                       [&partialQ, &distanceSum](const T& dist) { return partialQ + (0.5 * dist / distanceSum); });
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level)> calcDistribution(const std::vector<T>& sqDistances, const T& distanceSum)
    {
        T partialQ = 0.5 * (1.0 / static_cast<T>(sqDistances.size()));

#pragma omp parallel for schedule(static), shared(partialQ)
        for (int32_t i = 0; i < static_cast<int32_t>(sqDistances.size()); ++i)
        {
            m_distribution[i] = partialQ + (0.5 * sqDistances[i] / distanceSum);
        }
    }

    Coreset<T> sampleDistribution(const Matrix<T>* const data)
    {
        Coreset<T> coreset(m_sampleSize, data->cols());

        auto selectedIdxs = this->m_weightedSelector.selectMultiple(&m_distribution, m_sampleSize);
        for (const auto& idx : selectedIdxs)
        {
            coreset.append(data->crowBegin(idx), data->crowEnd(idx), 1.0 / (m_sampleSize * m_distribution[idx]));
        }

        return coreset;
    }

private:
    const Matrix<T>* p_data;
    int32_t m_sampleSize;
    std::vector<T> m_distribution;
};
}  // namespace hpkmeans