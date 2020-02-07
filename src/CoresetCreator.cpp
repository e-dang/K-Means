#include "CoresetCreator.hpp"

#include "omp.h"

void AbstractCoresetCreator::createCoreset(const Matrix* const data, Coreset* const coreset)
{
    std::vector<value_t> mean(data->getNumFeatures());
    std::vector<value_t> sqDistances(data->getNumData());
    std::vector<value_t> distribution(data->getNumData(), 0.0);

    calcMean(data, &mean);

    value_t distanceSum = calcDistsFromMean(data, &mean, &sqDistances);

    calcDistribution(&sqDistances, distanceSum, &distribution);

    sampleDistribution(data, &distribution, coreset);
}

void SharedMemoryCoresetCreator::calcMean(const Matrix* const data, std::vector<value_t>* const mean)
{
    pAverager->calculateAverage(data, mean);
}

value_t SharedMemoryCoresetCreator::calcDistsFromMean(const Matrix* const data, const std::vector<value_t>* const mean,
                                                      std::vector<value_t>* const sqDistances)
{
    return pDistSumCalc->calcDistances(data, mean, sqDistances, pDistanceFunc);
}

void SharedMemoryCoresetCreator::calcDistribution(const std::vector<value_t>* const sqDistances,
                                                  const value_t& distanceSum, std::vector<value_t>* const distribution)
{
    pDistrCalc->calcDistribution(sqDistances, distanceSum, distribution);
}

void SharedMemoryCoresetCreator::sampleDistribution(const Matrix* const data,
                                                    const std::vector<value_t>* const distribution,
                                                    Coreset* const coreset)
{
    auto selectedIdxs = pSelector->select(distribution, mSampleSize);
    for (auto& idx : selectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (mSampleSize * distribution->at(idx)));
    }
}

void MPICoresetCreator::calcMean(const Matrix* const data, std::vector<value_t>* const mean)
{
    chunkMeans    = Matrix(mNumProcs, data->getNumFeatures());
    mDistanceSums = std::vector<value_t>(data->getNumData());
    chunkMeans.resize(mNumProcs);
    setTotalNumData(getTotalNumDataMPI(data));

    pAverager->calculateSum(data, mean);
    MPI_Gather(mean->data(), mean->size(), MPI_FLOAT, chunkMeans.data(), mean->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
        std::fill(mean->begin(), mean->end(), 0.0);
        auto numData = std::accumulate(mLengths.begin(), mLengths.end(), (unsigned long long)0);
        pAverager->calculateSum(&chunkMeans, mean);
        pAverager->normalizeSum(mean, numData);
    }

    MPI_Bcast(mean->data(), mean->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

value_t MPICoresetCreator::calcDistsFromMean(const Matrix* const data, const std::vector<value_t>* const mean,
                                             std::vector<value_t>* const sqDistances)
{
    // calculate local quantization errors
    value_t localDistanceSum = pDistSumCalc->calcDistances(data, mean, sqDistances, pDistanceFunc);

    MPI_Gather(&localDistanceSum, 1, MPI_FLOAT, mDistanceSums.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&localDistanceSum, &mTotalDistanceSum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    return localDistanceSum;
}

void MPICoresetCreator::calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                                         std::vector<value_t>* const distribution)
{
    value_t totalDistanceSums;
    std::vector<int> uniformSampleCounts(mNumProcs, 0);
    std::vector<int> nonUniformSampleCounts(mNumProcs, 0);

    if (mRank == 0)
    {
        totalDistanceSums = std::accumulate(mDistanceSums.begin(), mDistanceSums.end(), 0.0);
        calculateSamplingStrategy(&uniformSampleCounts, &nonUniformSampleCounts, totalDistanceSums);
    }

    MPI_Bcast(&totalDistanceSums, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(uniformSampleCounts.data(), 1, MPI_INT, &mNumUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nonUniformSampleCounts.data(), 1, MPI_INT, &mNumNonUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::transform(sqDistances->begin(), sqDistances->end(), distribution->begin(),
                   [&totalDistanceSums](const value_t& dist) { return dist / totalDistanceSums; });
}

void MPICoresetCreator::sampleDistribution(const Matrix* const data, const std::vector<value_t>* const distribution,
                                           Coreset* const coreset)
{
    std::vector<value_t> uniformWeights(distribution->size(), 1.0 / mTotalNumData);

    appendDataToCoreset(data, coreset, &uniformWeights, distribution, mNumUniformSamples);
    appendDataToCoreset(data, coreset, distribution, distribution, mNumNonUniformSamples);

    distributeCoreset(coreset);
}

void MPICoresetCreator::appendDataToCoreset(const Matrix* const data, Coreset* const coreset,
                                            const std::vector<value_t>* const weights,
                                            const std::vector<value_t>* const distribution, const int& numSamples)
{
    value_t partialQ         = 0.5 * (1.0 / mTotalNumData);
    auto uniformSelectedIdxs = pSelector->select(weights, numSamples);
    for (const auto& idx : uniformSelectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.emplace_back(1.0 / (mSampleSize * (partialQ + 0.5 * distribution->at(idx))));
    }
}

void MPICoresetCreator::calculateSamplingStrategy(std::vector<int>* const uniformSampleCounts,
                                                  std::vector<int>* const nonUniformSampleCounts,
                                                  const value_t& totalDistanceSums)
{
    for (auto i = 0; i < mSampleSize; i++)
    {
        double randNum = getRandDouble01MPI();
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

void MPICoresetCreator::updateUniformSampleCounts(std::vector<int>* const uniformSampleCounts)
{
    double randNum              = getRandDouble01MPI() * mTotalNumData;
    unsigned long cumulativeSum = 0;
    for (auto j = 0; j < mNumProcs; j++)
    {
        cumulativeSum += mLengths[j];
        if (cumulativeSum >= randNum)
        {
            uniformSampleCounts->at(j)++;
            break;
        }
    }
}

void MPICoresetCreator::updateNonUniformSampleCounts(std::vector<int>* const nonUniformSampleCounts,
                                                     const value_t& totalDistanceSums)
{
    double randNum        = getRandDouble01MPI() * totalDistanceSums;
    value_t cumulativeSum = 0;
    for (auto j = 0; j < mNumProcs; j++)
    {
        cumulativeSum += mDistanceSums.at(j);
        if (cumulativeSum >= randNum)
        {
            nonUniformSampleCounts->at(j)++;
            break;
        }
    }
}

void MPICoresetCreator::distributeCoreset(Coreset* const coreset)
{
    // get the number of datapoints in each process' coreset
    auto numCoresetData = coreset->weights.size();
    std::vector<int> numCoresetDataPerProc(mNumProcs);
    MPI_Allgather(&numCoresetData, 1, MPI_INT, numCoresetDataPerProc.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // create length and displacement vectors for transfer of coreset data
    std::vector<int> matrixLengths(mNumProcs);
    std::vector<int> matrixDisplacements(mNumProcs, 0);
    std::vector<int> vectorDisplacements(mNumProcs, 0);
    for (auto i = 0; i < mNumProcs; i++)
    {
        matrixLengths.at(i) = numCoresetDataPerProc.at(i) * coreset->data.getNumFeatures();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + matrixLengths.at(i - 1);
            vectorDisplacements.at(i) = vectorDisplacements.at(i - 1) + numCoresetDataPerProc.at(i - 1);
        }
    }

    // create and fill temporary coreset with data at root
    Coreset fullCoreset{ Matrix(mSampleSize, coreset->data.getNumFeatures()), std::vector<value_t>(mSampleSize) };
    fullCoreset.data.resize(mSampleSize);
    MPI_Gatherv(coreset->data.data(), coreset->data.size(), MPI_FLOAT, fullCoreset.data.data(), matrixLengths.data(),
                matrixDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(coreset->weights.data(), coreset->weights.size(), MPI_FLOAT, fullCoreset.weights.data(),
                numCoresetDataPerProc.data(), vectorDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // get lengths and displacements for evenly distributing coreset data amoung processes
    auto mpiData        = getMPIData(mSampleSize);
    auto& vectorLengths = mpiData.lengths;
    vectorDisplacements = mpiData.displacements;
    for (auto i = 0; i < mNumProcs; i++)
    {
        matrixLengths.at(i) = vectorLengths.at(i) * coreset->data.getNumFeatures();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + matrixLengths.at(i - 1);
        }
    }

    // resize and distribute coreset data
    coreset->data.resize(vectorLengths.at(mRank));
    coreset->weights.resize(vectorLengths.at(mRank));
    MPI_Scatterv(fullCoreset.weights.data(), vectorLengths.data(), vectorDisplacements.data(), MPI_FLOAT,
                 coreset->weights.data(), coreset->weights.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullCoreset.data.data(), matrixLengths.data(), matrixDisplacements.data(), MPI_FLOAT,
                 coreset->data.data(), matrixLengths.at(mRank), MPI_FLOAT, 0, MPI_COMM_WORLD);
}