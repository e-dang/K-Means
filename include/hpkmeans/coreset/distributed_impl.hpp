#pragma once

#include <hpkmeans/coreset/coreset_creator_impl.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class DistributedCoresetCreatorImpl : public AbstractCoresetCreatorImpl<T, Level, DistanceFunc>
{
public:
    DistributedCoresetCreatorImpl() :
        AbstractCoresetCreatorImpl<T, Level, DistanceFunc>(),
        p_data(nullptr),
        m_sampleSize(-1),
        m_numUniformSamples(-1),
        m_numNonUniformSamples(-1),
        m_distanceSums(),
        m_distribution()
    {
    }

    Coreset<T> createCoreset(const Matrix<T>* const data, const int32_t& sampleSize) override
    {
        if (p_data != data || sampleSize != m_sampleSize)
        {
            setState(data, sampleSize);
            auto mean        = calcMean(data);
            auto sqDistances = calcDistsFromMean(data, mean);
            calcDistribution(sqDistances);
        }

        return sampleDistribution(data);
    }

private:
    void setState(const Matrix<T>* data, const int32_t& sampleSize)
    {
        p_data         = data;
        m_chunkifier   = Chunkifier<Level>(data->numRows());
        m_sampleSize   = sampleSize;
        m_distanceSums = std::vector<T>(m_chunkifier.numProcs(), 0.0);
        m_distribution = std::vector<T>(data->numRows(), 0.0);
    }

    std::vector<T> calcMean(const Matrix<T>* const data)
    {
        auto mean = this->calcMeanSum(data);
        MPI_Allreduce(MPI_IN_PLACE, mean.data(), mean.size(), matchMPIType<T>(), MPI_SUM, MPI_COMM_WORLD);
        auto totalNumData = static_cast<T>(m_chunkifier.totalNumData());
        std::for_each(mean.begin(), mean.end(), [&totalNumData](T& val) { val /= totalNumData; });

        return mean;
    }

    std::vector<T> calcDistsFromMean(const Matrix<T>* const data, const std::vector<T>& mean)
    {
        std::vector<T> sqDistances(data->numRows(), 0.0);

        auto localDistanceSum =
          AbstractCoresetCreatorImpl<T, Level, DistanceFunc>::calcDistsFromMean(data, mean, sqDistances);

        MPI_Gather(&localDistanceSum, 1, matchMPIType<T>(), m_distanceSums.data(), 1, matchMPIType<T>(), 0,
                   MPI_COMM_WORLD);

        return sqDistances;
    }

    void calcDistribution(const std::vector<T>& sqDistances)
    {
        T totalDistanceSums;
        std::vector<int32_t> uniformSampleCounts(m_chunkifier.numProcs(), 0);
        std::vector<int32_t> nonUniformSampleCounts(m_chunkifier.numProcs(), 0);

        if (getCommRank() == 0)
        {
            totalDistanceSums = std::accumulate(m_distanceSums.begin(), m_distanceSums.end(), 0.0);
            calculateSamplingStrategy(uniformSampleCounts, nonUniformSampleCounts, totalDistanceSums);
        }

        MPI_Bcast(&totalDistanceSums, 1, matchMPIType<T>(), 0, MPI_COMM_WORLD);
        MPI_Scatter(uniformSampleCounts.data(), 1, MPI_INT, &m_numUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(nonUniformSampleCounts.data(), 1, MPI_INT, &m_numNonUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::transform(sqDistances.cbegin(), sqDistances.cend(), m_distribution.begin(),
                       [&totalDistanceSums](const T& dist) { return dist / totalDistanceSums; });
    }

    void calculateSamplingStrategy(std::vector<int32_t>& uniformSampleCounts,
                                   std::vector<int32_t>& nonUniformSampleCounts, const T& totalDistanceSums)
    {
        for (int32_t i = 0; i < m_sampleSize; ++i)
        {
            auto randNum = getRandFraction();
            if (randNum >= 0.5)
                updateUniformSampleCounts(uniformSampleCounts);
            else
                updateNonUniformSampleCounts(nonUniformSampleCounts, totalDistanceSums);
        }
    }

    void updateUniformSampleCounts(std::vector<int32_t>& uniformSampleCounts)
    {
        auto randNum          = getRandFraction() * m_chunkifier.totalNumData();
        int32_t cumulativeSum = 0;
        for (int j = 0; j < m_chunkifier.numProcs(); ++j)
        {
            cumulativeSum += m_chunkifier.lengthsAt(j);
            if (cumulativeSum >= randNum)
            {
                ++uniformSampleCounts[j];
                break;
            }
        }
    }

    void updateNonUniformSampleCounts(std::vector<int32_t>& nonUniformSampleCounts, const T& totalDistanceSums)
    {
        auto randNum    = getRandFraction() * totalDistanceSums;
        T cumulativeSum = 0.0;
        for (int j = 0; j < m_chunkifier.numProcs(); ++j)
        {
            cumulativeSum += m_distanceSums[j];
            if (cumulativeSum >= randNum)
            {
                ++nonUniformSampleCounts[j];
                break;
            }
        }
    }

    Coreset<T> sampleDistribution(const Matrix<T>* const data)
    {
        Coreset<T> coreset(m_sampleSize, data->cols());
        std::vector<T> uniformWeights(m_distribution.size(), 1.0 / m_chunkifier.totalNumData());

        appendDataToCoreset(data, coreset, uniformWeights, m_numUniformSamples);
        appendDataToCoreset(data, coreset, m_distribution, m_numNonUniformSamples);
        coreset = distributeCoreset(data, coreset);

        return coreset;
    }

    void appendDataToCoreset(const Matrix<T>* const data, Coreset<T>& coreset, const std::vector<T>& weights,
                             const int32_t& numSamples)
    {
        T partialQ        = 0.5 * (1.0 / static_cast<T>(m_chunkifier.totalNumData()));
        auto selectedIdxs = this->m_weightedSelector.selectMultiple(&weights, numSamples);
        for (const auto& idx : selectedIdxs)
        {
            coreset.append(data->crowBegin(idx), data->crowEnd(idx),
                           1.0 / (m_sampleSize * (partialQ + 0.5 * m_distribution[idx])));
        }
    }

    Coreset<T> distributeCoreset(const Matrix<T>* const data, Coreset<T>& coreset)
    {
        // get the number of datapoints in each process' coreset
        auto numCoresetData = coreset.numRows();
        std::vector<int32_t> numCoresetDataPerProc(m_chunkifier.numProcs(), 0);
        MPI_Allgather(&numCoresetData, 1, MPI_INT, numCoresetDataPerProc.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // create length and displacement vectors for transfer of coreset data
        std::vector<int32_t> matrixLengths(m_chunkifier.numProcs());
        std::vector<int32_t> matrixDisplacements(m_chunkifier.numProcs(), 0);
        std::vector<int32_t> vectorDisplacements(m_chunkifier.numProcs(), 0);
        for (int i = 0; i < m_chunkifier.numProcs(); ++i)
        {
            matrixLengths[i] = numCoresetDataPerProc[i] * data->cols();
            if (i != 0)
            {
                matrixDisplacements[i] = matrixDisplacements[i - 1] + matrixLengths[i - 1];
                vectorDisplacements[i] = vectorDisplacements[i - 1] + numCoresetDataPerProc[i - 1];
            }
        }

        // create and fill temporary coreset with data at root
        Coreset<T> fullCoreset(m_sampleSize, data->cols(), true);

        MPI_Gatherv(coreset.data(), coreset.size(), matchMPIType<T>(), fullCoreset.data(), matrixLengths.data(),
                    matrixDisplacements.data(), matchMPIType<T>(), 0, MPI_COMM_WORLD);
        MPI_Gatherv(coreset.weights(), coreset.numRows(), matchMPIType<T>(), fullCoreset.weights(),
                    numCoresetDataPerProc.data(), vectorDisplacements.data(), matchMPIType<T>(), 0, MPI_COMM_WORLD);

        // get lengths and displacements for evenly distributing coreset data amoung processes
        Chunkifier<Level> mpiData(m_sampleSize, true);
        for (int i = 0; i < mpiData.numProcs(); ++i)
        {
            matrixLengths[i] = mpiData.lengthsAt(i) * data->cols();
            if (i != 0)
                matrixDisplacements[i] = matrixDisplacements[i - 1] + matrixLengths[i - 1];
        }

        // resize and distribute coreset data
        Coreset<T> newCoreset(mpiData.myLength(), data->cols(), true);

        MPI_Scatterv(fullCoreset.weights(), mpiData.lengths().data(), mpiData.displacements().data(), matchMPIType<T>(),
                     newCoreset.weights(), newCoreset.numRows(), matchMPIType<T>(), 0, MPI_COMM_WORLD);
        MPI_Scatterv(fullCoreset.data(), matrixLengths.data(), matrixDisplacements.data(), matchMPIType<T>(),
                     newCoreset.data(), matrixLengths[getCommRank()], matchMPIType<T>(), 0, MPI_COMM_WORLD);

        return newCoreset;
    }

private:
    Chunkifier<Level> m_chunkifier;
    const Matrix<T>* p_data;
    int32_t m_sampleSize;
    int32_t m_numUniformSamples;
    int32_t m_numNonUniformSamples;
    std::vector<T> m_distanceSums;
    std::vector<T> m_distribution;
};
}  // namespace hpkmeans