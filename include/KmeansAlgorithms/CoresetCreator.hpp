#pragma once

#include <memory>

#include "Strategies/Averager.hpp"
#include "Strategies/CoresetDistributionCalculator.hpp"
#include "Strategies/DistanceSumCalculator.hpp"
#include "Strategies/RandomSelector.hpp"
#include "Utils/Utils.hpp"
#include "omp.h"
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class AbstractCoresetCreator
{
protected:
    int_size mSampleSize;
    std::unique_ptr<IMultiWeightedRandomSelector<precision, int_size>> pSelector;
    std::unique_ptr<AbstractAverager<precision, int_size>> pAverager;
    std::unique_ptr<IDistanceSumCalculator<precision, int_size>> pDistSumCalc;
    std::shared_ptr<IDistanceFunctor<precision>> pDistanceFunc;

public:
    AbstractCoresetCreator(const int_size& sampleSize, IMultiWeightedRandomSelector<precision, int_size>* selector,
                           AbstractAverager<precision, int_size>* averager,
                           IDistanceSumCalculator<precision, int_size>* distSumCalc,
                           std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        mSampleSize(sampleSize),
        pSelector(selector),
        pAverager(averager),
        pDistSumCalc(distSumCalc),
        pDistanceFunc(distanceFunc)
    {
    }

    virtual ~AbstractCoresetCreator() = default;

    Coreset<precision, int_size> createCoreset(const Matrix<precision, int_size>* const data);

protected:
    virtual std::vector<precision> calcMean(const Matrix<precision, int_size>* const data) = 0;

    virtual precision calcDistsFromMean(const Matrix<precision, int_size>* const data,
                                        const std::vector<precision>* const mean,
                                        std::vector<precision>* const sqDistances) = 0;

    virtual std::vector<precision> calcDistribution(const std::vector<precision>* const sqDistances,
                                                    const precision& distanceSum) = 0;

    virtual Coreset<precision, int_size> sampleDistribution(const Matrix<precision, int_size>* const data,
                                                            const std::vector<precision>* const distribution) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SharedMemoryCoresetCreator : public AbstractCoresetCreator<precision, int_size>
{
protected:
    std::unique_ptr<ICoresetDistributionCalculator<precision, int_size>> pDistrCalc;

public:
    SharedMemoryCoresetCreator(const int_size& sampleSize, IMultiWeightedRandomSelector<precision, int_size>* selector,
                               AbstractAverager<precision, int_size>* averager,
                               IDistanceSumCalculator<precision, int_size>* distSumCalc,
                               ICoresetDistributionCalculator<precision, int_size>* distrCalc,
                               std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        AbstractCoresetCreator<precision, int_size>(sampleSize, selector, averager, distSumCalc, distanceFunc),
        pDistrCalc(distrCalc)
    {
    }

    ~SharedMemoryCoresetCreator() = default;

protected:
    std::vector<precision> calcMean(const Matrix<precision, int_size>* const data) override;

    precision calcDistsFromMean(const Matrix<precision, int_size>* const data, const std::vector<precision>* const mean,
                                std::vector<precision>* const sqDistances) override;

    std::vector<precision> calcDistribution(const std::vector<precision>* const sqDistances,
                                            const precision& distanceSum) override;

    Coreset<precision, int_size> sampleDistribution(const Matrix<precision, int_size>* const data,
                                                    const std::vector<precision>* const distribution) override;
};

template <typename precision = double, typename int_size = int32_t>
class MPICoresetCreator : public AbstractCoresetCreator<precision, int_size>
{
protected:
    int mRank;
    int mNumProcs;
    int_size mTotalNumData;
    int_size mNumUniformSamples;
    int_size mNumNonUniformSamples;
    precision mTotalDistanceSum;
    Matrix<precision, int_size> chunkMeans;
    std::vector<int_size> mLengths;
    std::vector<int_size> mDisplacements;
    std::vector<precision> mDistanceSums;

public:
    MPICoresetCreator(const int_size& sampleSize, IMultiWeightedRandomSelector<precision, int_size>* selector,
                      AbstractAverager<precision, int_size>* averager,
                      IDistanceSumCalculator<precision, int_size>* distSumCalc,
                      std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        AbstractCoresetCreator<precision, int_size>(sampleSize, selector, averager, distSumCalc, distanceFunc){};

    ~MPICoresetCreator() = default;

protected:
    void initMPIData(const int_size& totalNumData)
    {
        auto mpiData   = getMPIData(totalNumData);
        mRank          = mpiData.rank;
        mNumProcs      = mpiData.numProcs;
        mLengths       = mpiData.lengths;
        mDisplacements = mpiData.displacements;
        mTotalNumData  = totalNumData;
    }

    std::vector<precision> calcMean(const Matrix<precision, int_size>* const data) override;

    precision calcDistsFromMean(const Matrix<precision, int_size>* const data, const std::vector<precision>* const mean,
                                std::vector<precision>* const sqDistances) override;

    std::vector<precision> calcDistribution(const std::vector<precision>* const sqDistances,
                                            const precision& distanceSum) override;

    Coreset<precision, int_size> sampleDistribution(const Matrix<precision, int_size>* const data,
                                                    const std::vector<precision>* const distribution) override;

    void appendDataToCoreset(const Matrix<precision, int_size>* const data, Coreset<precision, int_size>* const coreset,
                             const std::vector<precision>* const weights,
                             const std::vector<precision>* const distribution, const int_size& numSamples);

    void calculateSamplingStrategy(std::vector<int_size>* const uniformSampleCounts,
                                   std::vector<int_size>* const nonUniformSampleCounts,
                                   const precision& totalDistanceSums);

    void updateUniformSampleCounts(std::vector<int_size>* const uniformSampleCounts);

    void updateNonUniformSampleCounts(std::vector<int_size>* const nonUniformSampleCounts,
                                      const precision& totalDistanceSums);

    void distributeCoreset(Coreset<precision, int_size>* const coreset);
};

template <typename precision, typename int_size>
Coreset<precision, int_size> AbstractCoresetCreator<precision, int_size>::createCoreset(
  const Matrix<precision, int_size>* const data)
{
    auto mean = calcMean(data);

    std::vector<precision> sqDistances(data->size());
    precision distanceSum = calcDistsFromMean(data, &mean, &sqDistances);

    auto distribution = calcDistribution(&sqDistances, distanceSum);

    return sampleDistribution(data, &distribution);
}

template <typename precision, typename int_size>
std::vector<precision> SharedMemoryCoresetCreator<precision, int_size>::calcMean(
  const Matrix<precision, int_size>* const data)
{
    std::vector<precision> mean(data->cols());
    this->pAverager->calculateAverage(data, &mean);
    return mean;
}

template <typename precision, typename int_size>
precision SharedMemoryCoresetCreator<precision, int_size>::calcDistsFromMean(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const mean,
  std::vector<precision>* const sqDistances)
{
    return this->pDistSumCalc->calcDistances(data, mean, sqDistances, this->pDistanceFunc);
}

template <typename precision, typename int_size>
std::vector<precision> SharedMemoryCoresetCreator<precision, int_size>::calcDistribution(
  const std::vector<precision>* const sqDistances, const precision& distanceSum)
{
    std::vector<precision> distribution(sqDistances->size(), 0.0);
    pDistrCalc->calcDistribution(sqDistances, distanceSum, &distribution);
    return distribution;
}

template <typename precision, typename int_size>
Coreset<precision, int_size> SharedMemoryCoresetCreator<precision, int_size>::sampleDistribution(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const distribution)
{
    Coreset<precision, int_size> coreset(this->mSampleSize, data->cols(), false);

    auto selectedIdxs = this->pSelector->select(distribution, this->mSampleSize);
    for (const auto& idx : selectedIdxs)
    {
        coreset.data.push_back(data->at(idx));
        coreset.weights.emplace_back(1.0 / (this->mSampleSize * distribution->at(idx)));
    }

    return coreset;
}

template <typename precision, typename int_size>
std::vector<precision> MPICoresetCreator<precision, int_size>::calcMean(const Matrix<precision, int_size>* const data)
{
    initMPIData(getTotalNumDataMPI(data));
    std::vector<precision> mean(data->cols());
    chunkMeans    = Matrix<precision, int_size>(mNumProcs, data->cols());
    mDistanceSums = std::vector<precision>(data->size());

    this->pAverager->calculateSum(data, &mean);
    MPI_Gather(mean.data(), mean.size(), mpi_type_t, chunkMeans.data(), mean.size(), mpi_type_t, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
        std::fill(mean.begin(), mean.end(), 0.0);
        auto numData = std::accumulate(mLengths.begin(), mLengths.end(), 0);
        this->pAverager->calculateSum(&chunkMeans, &mean);
        this->pAverager->normalizeSum(&mean, numData);
    }

    MPI_Bcast(mean.data(), mean.size(), mpi_type_t, 0, MPI_COMM_WORLD);

    return mean;
}
template <typename precision, typename int_size>
precision MPICoresetCreator<precision, int_size>::calcDistsFromMean(const Matrix<precision, int_size>* const data,
                                                                    const std::vector<precision>* const mean,
                                                                    std::vector<precision>* const sqDistances)
{
    // calculate local quantization errors
    precision localDistanceSum = this->pDistSumCalc->calcDistances(data, mean, sqDistances, this->pDistanceFunc);

    MPI_Gather(&localDistanceSum, 1, mpi_type_t, mDistanceSums.data(), 1, mpi_type_t, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&localDistanceSum, &mTotalDistanceSum, 1, mpi_type_t, MPI_SUM, MPI_COMM_WORLD);

    return localDistanceSum;
}
template <typename precision, typename int_size>
std::vector<precision> MPICoresetCreator<precision, int_size>::calcDistribution(
  const std::vector<precision>* const sqDistances, const precision& distanceSum)
{
    precision totalDistanceSums;
    std::vector<precision> distribution(sqDistances->size(), 0.0);
    std::vector<int32_t> uniformSampleCounts(mNumProcs, 0);
    std::vector<int32_t> nonUniformSampleCounts(mNumProcs, 0);

    if (mRank == 0)
    {
        totalDistanceSums = std::accumulate(mDistanceSums.begin(), mDistanceSums.end(), 0.0);
        calculateSamplingStrategy(&uniformSampleCounts, &nonUniformSampleCounts, totalDistanceSums);
    }

    MPI_Bcast(&totalDistanceSums, 1, mpi_type_t, 0, MPI_COMM_WORLD);
    MPI_Scatter(uniformSampleCounts.data(), 1, MPI_INT, &mNumUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nonUniformSampleCounts.data(), 1, MPI_INT, &mNumNonUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::transform(sqDistances->begin(), sqDistances->end(), distribution.begin(),
                   [&totalDistanceSums](const precision& dist) { return dist / totalDistanceSums; });

    return distribution;
}
template <typename precision, typename int_size>
Coreset<precision, int_size> MPICoresetCreator<precision, int_size>::sampleDistribution(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const distribution)
{
    Coreset<precision, int_size> coreset(this->mSampleSize, data->cols(), false);
    std::vector<precision> uniformWeights(distribution->size(), 1.0 / mTotalNumData);

    appendDataToCoreset(data, &coreset, &uniformWeights, distribution, mNumUniformSamples);
    appendDataToCoreset(data, &coreset, distribution, distribution, mNumNonUniformSamples);

    distributeCoreset(&coreset);

    return coreset;
}
template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::appendDataToCoreset(const Matrix<precision, int_size>* const data,
                                                                 Coreset<precision, int_size>* const coreset,
                                                                 const std::vector<precision>* const weights,
                                                                 const std::vector<precision>* const distribution,
                                                                 const int_size& numSamples)
{
    precision partialQ       = 0.5 * (1.0 / mTotalNumData);
    auto uniformSelectedIdxs = this->pSelector->select(weights, numSamples);
    for (const auto& idx : uniformSelectedIdxs)
    {
        coreset->data.push_back(data->at(idx));
        coreset->weights.emplace_back(1.0 / (this->mSampleSize * (partialQ + 0.5 * distribution->at(idx))));
    }
}
template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::calculateSamplingStrategy(
  std::vector<int_size>* const uniformSampleCounts, std::vector<int_size>* const nonUniformSampleCounts,
  const precision& totalDistanceSums)
{
    for (int_size i = 0; i < this->mSampleSize; ++i)
    {
        precision randNum = getRandDouble01MPI();
        if (randNum >= 0.5)
        {
            updateUniformSampleCounts(uniformSampleCounts);
        }
        else
        {
            updateNonUniformSampleCounts(nonUniformSampleCounts, totalDistanceSums);
        }
    }
}
template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::updateUniformSampleCounts(std::vector<int_size>* const uniformSampleCounts)
{
    precision randNum      = getRandDouble01MPI() * mTotalNumData;
    int_size cumulativeSum = 0;
    for (int j = 0; j < mNumProcs; ++j)
    {
        cumulativeSum += mLengths[j];
        if (cumulativeSum >= randNum)
        {
            ++uniformSampleCounts->at(j);
            break;
        }
    }
}
template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::updateNonUniformSampleCounts(
  std::vector<int_size>* const nonUniformSampleCounts, const precision& totalDistanceSums)
{
    precision randNum       = getRandDouble01MPI() * totalDistanceSums;
    precision cumulativeSum = 0;
    for (int j = 0; j < mNumProcs; ++j)
    {
        cumulativeSum += mDistanceSums.at(j);
        if (cumulativeSum >= randNum)
        {
            ++nonUniformSampleCounts->at(j);
            break;
        }
    }
}
template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::distributeCoreset(Coreset<precision, int_size>* const coreset)
{
    // get the number of datapoints in each process' coreset
    auto numCoresetData = coreset->weights.size();
    std::vector<int_size> numCoresetDataPerProc(mNumProcs);
    MPI_Allgather(&numCoresetData, 1, MPI_INT, numCoresetDataPerProc.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // create length and displacement vectors for transfer of coreset data
    std::vector<int_size> matrixLengths(mNumProcs);
    std::vector<int_size> matrixDisplacements(mNumProcs, 0);
    std::vector<int_size> vectorDisplacements(mNumProcs, 0);
    for (int i = 0; i < mNumProcs; ++i)
    {
        matrixLengths.at(i) = numCoresetDataPerProc.at(i) * coreset->data.cols();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + matrixLengths.at(i - 1);
            vectorDisplacements.at(i) = vectorDisplacements.at(i - 1) + numCoresetDataPerProc.at(i - 1);
        }
    }

    // create and fill temporary coreset with data at root
    Coreset<precision, int_size> fullCoreset(this->mSampleSize, coreset->data.cols());

    MPI_Gatherv(coreset->data.data(), coreset->data.elements(), mpi_type_t, fullCoreset.data.data(),
                matrixLengths.data(), matrixDisplacements.data(), mpi_type_t, 0, MPI_COMM_WORLD);
    MPI_Gatherv(coreset->weights.data(), coreset->weights.size(), mpi_type_t, fullCoreset.weights.data(),
                numCoresetDataPerProc.data(), vectorDisplacements.data(), mpi_type_t, 0, MPI_COMM_WORLD);

    // get lengths and displacements for evenly distributing coreset data amoung processes
    auto mpiData        = getMPIData(this->mSampleSize);
    auto& vectorLengths = mpiData.lengths;
    vectorDisplacements = mpiData.displacements;
    for (int32_t i = 0; i < mNumProcs; ++i)
    {
        matrixLengths.at(i) = vectorLengths.at(i) * coreset->data.cols();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + matrixLengths.at(i - 1);
        }
    }

    // resize and distribute coreset data
    auto numFeatures = coreset->data.cols();
    *coreset         = Coreset<precision, int_size>(vectorLengths.at(mRank), numFeatures);

    MPI_Scatterv(fullCoreset.weights.data(), vectorLengths.data(), vectorDisplacements.data(), mpi_type_t,
                 coreset->weights.data(), coreset->weights.size(), mpi_type_t, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullCoreset.data.data(), matrixLengths.data(), matrixDisplacements.data(), mpi_type_t,
                 coreset->data.data(), matrixLengths.at(mRank), mpi_type_t, 0, MPI_COMM_WORLD);
}
}  // namespace HPKmeans