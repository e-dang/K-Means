#pragma once

#include <omp.h>

#include <hpkmeans/algorithms/kmeans_algorithm.hpp>
#include <hpkmeans/algorithms/strategies/averager.hpp>
#include <hpkmeans/algorithms/strategies/coreset_distribution_calculator.hpp>
#include <hpkmeans/algorithms/strategies/distance_sum_calculator.hpp>
#include <hpkmeans/algorithms/strategies/random_selector.hpp>
#include <hpkmeans/data_types/coreset.hpp>
#include <hpkmeans/data_types/data_chunks.hpp>
#include <hpkmeans/utils/utils.hpp>
#include <hpkmeans/utils/mpi_class.hpp>
#include <memory>

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractCoresetCreator : public AbstractKmeansAlgorithm<precision, int_size>
{
protected:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;

    int_size mSampleSize;
    std::unique_ptr<IMultiWeightedRandomSelector<precision, int_size>> p_Selector;
    std::unique_ptr<AbstractAverager<precision, int_size>> p_Averager;
    std::unique_ptr<IDistanceSumCalculator<precision, int_size>> p_DistSumCalc;

public:
    AbstractCoresetCreator(const int_size& sampleSize, IMultiWeightedRandomSelector<precision, int_size>* selector,
                           AbstractAverager<precision, int_size>* averager,
                           IDistanceSumCalculator<precision, int_size>* distSumCalc) :
        mSampleSize(sampleSize), p_Selector(selector), p_Averager(averager), p_DistSumCalc(distSumCalc)
    {
    }

    virtual ~AbstractCoresetCreator() = default;

    Coreset<precision, int_size> createCoreset();

protected:
    virtual std::vector<precision> calcMean() = 0;

    virtual precision calcDistsFromMean(const std::vector<precision>* const mean,
                                        std::vector<precision>* const sqDistances) = 0;

    virtual std::vector<precision> calcDistribution(const std::vector<precision>* const sqDistances,
                                                    const precision& distanceSum) = 0;

    virtual Coreset<precision, int_size> sampleDistribution(const std::vector<precision>* const distribution) = 0;
};

template <typename precision, typename int_size>
class SharedMemoryCoresetCreator : public AbstractCoresetCreator<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;

protected:
    std::unique_ptr<ICoresetDistributionCalculator<precision, int_size>> pDistrCalc;

public:
    SharedMemoryCoresetCreator(const int_size& sampleSize, IMultiWeightedRandomSelector<precision, int_size>* selector,
                               AbstractAverager<precision, int_size>* averager,
                               IDistanceSumCalculator<precision, int_size>* distSumCalc,
                               ICoresetDistributionCalculator<precision, int_size>* distrCalc) :
        AbstractCoresetCreator<precision, int_size>(sampleSize, selector, averager, distSumCalc), pDistrCalc(distrCalc)
    {
    }

    ~SharedMemoryCoresetCreator() = default;

protected:
    std::vector<precision> calcMean() override;

    precision calcDistsFromMean(const std::vector<precision>* const mean,
                                std::vector<precision>* const sqDistances) override;

    std::vector<precision> calcDistribution(const std::vector<precision>* const sqDistances,
                                            const precision& distanceSum) override;

    Coreset<precision, int_size> sampleDistribution(const std::vector<precision>* const distribution) override;
};

template <typename precision, typename int_size>
class MPICoresetCreator : public AbstractCoresetCreator<precision, int_size>, public MPIClass<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;
    using MPIClass<precision, int_size>::mpi_precision;
    using MPIClass<precision, int_size>::mpi_int_size;

protected:
    int_size mNumUniformSamples;
    int_size mNumNonUniformSamples;
    std::vector<precision> mDistanceSums;

public:
    MPICoresetCreator(const int_size& sampleSize, IMultiWeightedRandomSelector<precision, int_size>* selector,
                      AbstractAverager<precision, int_size>* averager,
                      IDistanceSumCalculator<precision, int_size>* distSumCalc) :
        AbstractCoresetCreator<precision, int_size>(sampleSize, selector, averager, distSumCalc)
    {
    }

    ~MPICoresetCreator() = default;

protected:
    std::vector<precision> calcMean() override;

    precision calcDistsFromMean(const std::vector<precision>* const mean,
                                std::vector<precision>* const sqDistances) override;

    std::vector<precision> calcDistribution(const std::vector<precision>* const sqDistances,
                                            const precision& distanceSum) override;

    Coreset<precision, int_size> sampleDistribution(const std::vector<precision>* const distribution) override;

    void appendDataToCoreset(Coreset<precision, int_size>* const coreset, const std::vector<precision>* const weights,
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
Coreset<precision, int_size> AbstractCoresetCreator<precision, int_size>::createCoreset()
{
    auto mean = calcMean();

    std::vector<precision> sqDistances(p_KmeansState->dataSize());
    precision distanceSum = calcDistsFromMean(&mean, &sqDistances);

    auto distribution = calcDistribution(&sqDistances, distanceSum);

    return sampleDistribution(&distribution);
}

template <typename precision, typename int_size>
std::vector<precision> SharedMemoryCoresetCreator<precision, int_size>::calcMean()
{
    std::vector<precision> mean(p_KmeansState->dataCols());
    this->p_Averager->calculateAverage(p_KmeansState->data(), &mean);
    return mean;
}

template <typename precision, typename int_size>
precision SharedMemoryCoresetCreator<precision, int_size>::calcDistsFromMean(const std::vector<precision>* const mean,
                                                                             std::vector<precision>* const sqDistances)
{
    return this->p_DistSumCalc->calcDistances(p_KmeansState, mean, sqDistances);
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
  const std::vector<precision>* const distribution)
{
    Coreset<precision, int_size> coreset(this->mSampleSize, p_KmeansState->dataCols());

    auto selectedIdxs = this->p_Selector->select(distribution, this->mSampleSize);
    for (const auto& idx : selectedIdxs)
    {
        coreset.data.push_back(p_KmeansState->dataAt(idx));
        coreset.weights.emplace_back(1.0 / (this->mSampleSize * distribution->at(idx)));
    }

    return coreset;
}

template <typename precision, typename int_size>
std::vector<precision> MPICoresetCreator<precision, int_size>::calcMean()
{
    std::vector<precision> mean(p_KmeansState->dataCols());
    Matrix<precision, int_size> chunkMeans(p_KmeansState->numProcs(), p_KmeansState->dataCols(), true);

    this->p_Averager->calculateSum(p_KmeansState->data(), &mean);

    MPI_Gather(mean.data(), mean.size(), mpi_precision, chunkMeans.data(), mean.size(), mpi_precision, 0,
               MPI_COMM_WORLD);

    if (p_KmeansState->rank() == 0)
    {
        std::fill(mean.begin(), mean.end(), 0.0);
        this->p_Averager->calculateSum(&chunkMeans, &mean);
        this->p_Averager->normalizeSum(&mean, p_KmeansState->totalNumData());
    }

    MPI_Bcast(mean.data(), mean.size(), mpi_precision, 0, MPI_COMM_WORLD);

    return mean;
}

template <typename precision, typename int_size>
precision MPICoresetCreator<precision, int_size>::calcDistsFromMean(const std::vector<precision>* const mean,
                                                                    std::vector<precision>* const sqDistances)
{
    // calculate local quantization errors
    precision localDistanceSum = this->p_DistSumCalc->calcDistances(p_KmeansState, mean, sqDistances);
    mDistanceSums              = std::vector<precision>(p_KmeansState->dataSize());

    MPI_Gather(&localDistanceSum, 1, mpi_precision, mDistanceSums.data(), 1, mpi_precision, 0, MPI_COMM_WORLD);

    return localDistanceSum;
}

template <typename precision, typename int_size>
std::vector<precision> MPICoresetCreator<precision, int_size>::calcDistribution(
  const std::vector<precision>* const sqDistances, const precision& distanceSum)
{
    precision totalDistanceSums;
    std::vector<precision> distribution(sqDistances->size(), 0.0);
    std::vector<int_size> uniformSampleCounts(p_KmeansState->numProcs(), 0);
    std::vector<int_size> nonUniformSampleCounts(p_KmeansState->numProcs(), 0);

    if (p_KmeansState->rank() == 0)
    {
        totalDistanceSums = std::accumulate(mDistanceSums.begin(), mDistanceSums.end(), 0.0);
        calculateSamplingStrategy(&uniformSampleCounts, &nonUniformSampleCounts, totalDistanceSums);
    }

    MPI_Bcast(&totalDistanceSums, 1, mpi_precision, 0, MPI_COMM_WORLD);
    MPI_Scatter(uniformSampleCounts.data(), 1, mpi_int_size, &mNumUniformSamples, 1, mpi_int_size, 0, MPI_COMM_WORLD);
    MPI_Scatter(nonUniformSampleCounts.data(), 1, mpi_int_size, &mNumNonUniformSamples, 1, mpi_int_size, 0,
                MPI_COMM_WORLD);

    std::transform(sqDistances->begin(), sqDistances->end(), distribution.begin(),
                   [&totalDistanceSums](const precision& dist) { return dist / totalDistanceSums; });

    return distribution;
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
    precision randNum      = getRandDouble01MPI() * p_KmeansState->totalNumData();
    int_size cumulativeSum = 0;
    for (int j = 0; j < p_KmeansState->numProcs(); ++j)
    {
        cumulativeSum += p_KmeansState->lengthsAt(j);
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
    precision cumulativeSum = 0.0;
    for (int j = 0; j < p_KmeansState->numProcs(); ++j)
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
Coreset<precision, int_size> MPICoresetCreator<precision, int_size>::sampleDistribution(
  const std::vector<precision>* const distribution)
{
    Coreset<precision, int_size> coreset(this->mSampleSize, p_KmeansState->dataCols());
    std::vector<precision> uniformWeights(distribution->size(), 1.0 / p_KmeansState->totalNumData());

    appendDataToCoreset(&coreset, &uniformWeights, distribution, mNumUniformSamples);
    appendDataToCoreset(&coreset, distribution, distribution, mNumNonUniformSamples);

    distributeCoreset(&coreset);

    return coreset;
}

template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::appendDataToCoreset(Coreset<precision, int_size>* const coreset,
                                                                 const std::vector<precision>* const weights,
                                                                 const std::vector<precision>* const distribution,
                                                                 const int_size& numSamples)
{
    precision partialQ = 0.5 * (1.0 / p_KmeansState->totalNumData());
    auto selectedIdxs  = this->p_Selector->select(weights, numSamples);
    for (const auto& idx : selectedIdxs)
    {
        coreset->data.push_back(p_KmeansState->dataAt(idx));
        coreset->weights.emplace_back(1.0 / (this->mSampleSize * (partialQ + 0.5 * distribution->at(idx))));
    }
}

template <typename precision, typename int_size>
void MPICoresetCreator<precision, int_size>::distributeCoreset(Coreset<precision, int_size>* const coreset)
{
    // get the number of datapoints in each process' coreset
    auto numCoresetData = coreset->weights.size();
    std::vector<int_size> numCoresetDataPerProc(p_KmeansState->numProcs());
    MPI_Allgather(&numCoresetData, 1, mpi_int_size, numCoresetDataPerProc.data(), 1, mpi_int_size, MPI_COMM_WORLD);

    // create length and displacement vectors for transfer of coreset data
    std::vector<int_size> matrixLengths(p_KmeansState->numProcs());
    std::vector<int_size> matrixDisplacements(p_KmeansState->numProcs(), 0);
    std::vector<int_size> vectorDisplacements(p_KmeansState->numProcs(), 0);
    for (int i = 0; i < p_KmeansState->numProcs(); ++i)
    {
        matrixLengths.at(i) = numCoresetDataPerProc.at(i) * p_KmeansState->dataCols();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + matrixLengths.at(i - 1);
            vectorDisplacements.at(i) = vectorDisplacements.at(i - 1) + numCoresetDataPerProc.at(i - 1);
        }
    }

    // create and fill temporary coreset with data at root
    Coreset<precision, int_size> fullCoreset(this->mSampleSize, p_KmeansState->dataCols(), true);

    MPI_Gatherv(coreset->data.data(), coreset->data.elements(), mpi_precision, fullCoreset.data.data(),
                matrixLengths.data(), matrixDisplacements.data(), mpi_precision, 0, MPI_COMM_WORLD);
    MPI_Gatherv(coreset->weights.data(), coreset->weights.size(), mpi_precision, fullCoreset.weights.data(),
                numCoresetDataPerProc.data(), vectorDisplacements.data(), mpi_precision, 0, MPI_COMM_WORLD);

    // get lengths and displacements for evenly distributing coreset data amoung processes
    MPIDataChunks<int_size> mpiData(this->mSampleSize);
    for (int32_t i = 0; i < mpiData.numProcs(); ++i)
    {
        matrixLengths.at(i) = mpiData.lengthsAt(i) * p_KmeansState->dataCols();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + matrixLengths.at(i - 1);
        }
    }

    // resize and distribute coreset data
    *coreset = Coreset<precision, int_size>(mpiData.myLength(), p_KmeansState->dataCols(), true);

    MPI_Scatterv(fullCoreset.weights.data(), mpiData.lengthsData(), mpiData.displacementsData(), mpi_precision,
                 coreset->weights.data(), coreset->weights.size(), mpi_precision, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullCoreset.data.data(), matrixLengths.data(), matrixDisplacements.data(), mpi_precision,
                 coreset->data.data(), matrixLengths.at(mpiData.rank()), mpi_precision, 0, MPI_COMM_WORLD);
}
}  // namespace HPKmeans