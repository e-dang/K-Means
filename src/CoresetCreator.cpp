#include "CoresetCreator.hpp"

#include "omp.h"

void AbstractCoresetCreator::createCoreset(Matrix* data, Coreset* coreset)
{
    std::vector<value_t> mean(data->getNumFeatures());
    std::vector<value_t> sqDistances(data->getNumData());
    std::vector<value_t> distribution(data->getNumData(), 0);

    calcMean(data, &mean);

    value_t distanceSum = calcDistsFromMean(data, &mean, &sqDistances);

    calcDistribution(&sqDistances, distanceSum, &distribution);

    sampleDistribution(data, &distribution, coreset);
}

void AbstractCoresetCreator::calcMean(Matrix* data, std::vector<value_t>* mean)
{
    pAverager->calculateAverage(data, mean);
}

value_t AbstractCoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                                  std::vector<value_t>* sqDistances)
{
    value_t distanceSum = 0;
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*pDistanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

void AbstractCoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                              std::vector<value_t>* distribution)
{
    value_t partialQ = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
    for (int i = 0; i < sqDistances->size(); i++)
    {
        distribution->at(i) = partialQ + 0.5 * sqDistances->at(i) / distanceSum;
    }
}

void AbstractCoresetCreator::sampleDistribution(Matrix* data, std::vector<value_t>* distribution, Coreset* coreset)
{
    auto selectedIdxs = pSelector->select(distribution, mSampleSize);
    for (auto& idx : selectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (mSampleSize * distribution->at(idx)));
    }
}

value_t OMPCoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                             std::vector<value_t>* sqDistances)
{
    value_t distanceSum = 0;
#pragma omp parallel for shared(data, mean, sqDistances, pDistanceFunc), schedule(static), reduction(+ : distanceSum)
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*pDistanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

void OMPCoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                         std::vector<value_t>* distribution)
{
    value_t partialQ = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
#pragma omp parallel for shared(sqDistances, distanceSum, distribution, partialQ), schedule(static)
    for (int i = 0; i < sqDistances->size(); i++)
    {
        distribution->at(i) = partialQ + 0.5 * sqDistances->at(i) / distanceSum;
    }
}

void MPICoresetCreator::calcMean(Matrix* data, std::vector<value_t>* mean)
{
    chunkMeans    = Matrix(mNumProcs, data->getNumFeatures());
    mDistanceSums = std::vector<value_t>(data->getNumData());
    chunkMeans.resize(mNumProcs);

    pAverager->calculateSum(data, mean);
    MPI_Gather(mean->data(), mean->size(), MPI_FLOAT, chunkMeans.data(), mean->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
        std::fill(mean->begin(), mean->end(), 0);
        int numData = std::accumulate(mLengths.begin(), mLengths.end(), 0);
        pAverager->calculateSum(&chunkMeans, mean);
        pAverager->normalizeSum(mean, numData);
    }

    MPI_Bcast(mean->data(), mean->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

value_t MPICoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                             std::vector<value_t>* sqDistances)
{
    // calculate local quantization errors
    value_t localDistanceSum = 0;
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) += std::pow((*pDistanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        localDistanceSum += sqDistances->at(i);
    }

    MPI_Gather(&localDistanceSum, 1, MPI_FLOAT, mDistanceSums.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&localDistanceSum, &mTotalDistanceSum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    return localDistanceSum;
}

void MPICoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                         std::vector<value_t>* distribution)
{
    value_t totalDistanceSums;
    std::vector<int> uniformSampleCounts(mNumProcs, 0);
    std::vector<int> nonUniformSampleCounts(mNumProcs, 0);

    if (mRank == 0)
    {
        totalDistanceSums = std::accumulate(mDistanceSums.begin(), mDistanceSums.end(), 0);
        calculateSamplingStrategy(&uniformSampleCounts, &nonUniformSampleCounts, totalDistanceSums);
    }

    MPI_Bcast(&totalDistanceSums, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(uniformSampleCounts.data(), 1, MPI_INT, &mNumUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nonUniformSampleCounts.data(), 1, MPI_INT, &mNumNonUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < distribution->size(); i++)
    {
        distribution->at(i) = sqDistances->at(i) / totalDistanceSums;
    }
}

void MPICoresetCreator::sampleDistribution(Matrix* data, std::vector<value_t>* distribution, Coreset* coreset)
{
    std::vector<value_t> uniformWeights(distribution->size(), 1.0 / mTotalNumData);

    appendDataToCoreset(data, coreset, &uniformWeights, distribution, mNumUniformSamples);
    appendDataToCoreset(data, coreset, distribution, distribution, mNumNonUniformSamples);

    distributeCoreset(coreset);
}

void MPICoresetCreator::appendDataToCoreset(Matrix* data, Coreset* coreset, std::vector<value_t>* weights,
                                            std::vector<value_t>* distribution, const int& numSamples)
{
    value_t partialQ         = 0.5 * (1.0 / mTotalNumData);
    auto uniformSelectedIdxs = pSelector->select(weights, numSamples);
    for (auto& idx : uniformSelectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (mSampleSize * (partialQ + 0.5 * distribution->at(idx))));
    }
}

void AbstractCoresetCreator::finishClustering(Matrix* data, ClusterResults* clusterResults)
{
    for (int i = 0; i < data->getNumData(); i++)
    {
        // auto closestCluster =
        //   pFinder->findClosestCluster(data->at(i), clusterResults->mClusterData.mClusters, ppDistanceFunc);

        int clusterIdx;
        value_t minDistance = -1;
        auto& clusters      = clusterResults->mClusterData.mClusters;
        for (int j = 0; j < clusters.getNumData(); j++)
        {
            value_t tempDistance = (*pDistanceFunc)(data->at(i), clusters.at(j), clusters.getNumFeatures());
            if (minDistance > tempDistance || minDistance < 0)
            {
                minDistance = tempDistance;
                clusterIdx  = j;
            }
        }

        clusterResults->mSqDistances.at(i)             = std::pow(minDistance, 2);
        clusterResults->mClusterData.mClustering.at(i) = clusterIdx;
    }
    clusterResults->mError =
      std::accumulate(clusterResults->mSqDistances.begin(), clusterResults->mSqDistances.end(), 0);
}

void MPICoresetCreator::finishClustering(Matrix* data, ClusterResults* clusterResults)
{
    for (int i = 0; i < data->getNumData(); i++)
    {
        // auto closestCluster =
        //   pFinder->findClosestCluster(data->at(i), clusterResults->mClusterData.mClusters, ppDistanceFunc);

        int clusterIdx;
        value_t minDistance = -1;
        auto& clusters      = clusterResults->mClusterData.mClusters;
        for (int j = 0; j < clusters.getNumData(); j++)
        {
            value_t tempDistance = (*pDistanceFunc)(data->at(i), clusters.at(j), clusters.getNumFeatures());
            if (minDistance > tempDistance || minDistance < 0)
            {
                minDistance = tempDistance;
                clusterIdx  = j;
            }
        }

        clusterResults->mSqDistances.at(mDisplacements.at(mRank) + i)             = std::pow(minDistance, 2);
        clusterResults->mClusterData.mClustering.at(mDisplacements.at(mRank) + i) = clusterIdx;
    }
    clusterResults->mError =
      std::accumulate(clusterResults->mSqDistances.begin(), clusterResults->mSqDistances.end(), 0);
    MPI_Allgatherv(MPI_IN_PLACE, mLengths.at(mRank), MPI_INT, clusterResults->mClusterData.mClustering.data(),
                   mLengths.data(), mDisplacements.data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, mLengths.at(mRank), MPI_FLOAT, clusterResults->mSqDistances.data(), mLengths.data(),
                   mDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &clusterResults->mError, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void MPICoresetCreator::calculateSamplingStrategy(std::vector<int>* uniformSampleCounts,
                                                  std::vector<int>* nonUniformSampleCounts,
                                                  const value_t& totalDistanceSums)
{
    for (int i = 0; i < mSampleSize; i++)
    {
        value_t randNum = getRandFloat01MPI();
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

void MPICoresetCreator::updateUniformSampleCounts(std::vector<int>* uniformSampleCounts)
{
    float randNum     = getRandFloat01MPI() * mTotalNumData;
    int cumulativeSum = 0;
    for (int j = 0; j < mNumProcs; j++)
    {
        cumulativeSum += mLengths[j];
        if (cumulativeSum >= randNum)
        {
            uniformSampleCounts->at(j)++;
            break;
        }
    }
}

void MPICoresetCreator::updateNonUniformSampleCounts(std::vector<int>* nonUniformSampleCounts,
                                                     const value_t& totalDistanceSums)
{
    float randNum         = getRandFloat01MPI() * totalDistanceSums;
    value_t cumulativeSum = 0;
    for (int j = 0; j < mNumProcs; j++)
    {
        cumulativeSum += mDistanceSums.at(j);
        if (cumulativeSum >= randNum)
        {
            nonUniformSampleCounts->at(j)++;
            break;
        }
    }
}

void MPICoresetCreator::distributeCoreset(Coreset* coreset)
{
    // get the number of datapoints in each process' coreset
    int numCoresetData = coreset->weights.size();
    std::vector<int> numCoresetDataPerProc(mNumProcs);
    MPI_Allgather(&numCoresetData, 1, MPI_INT, numCoresetDataPerProc.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // create length and displacement vectors for transfer of coreset data
    std::vector<int> matrixLengths(mNumProcs);
    std::vector<int> matrixDisplacements(mNumProcs, 0);
    std::vector<int> vectorDisplacements(mNumProcs, 0);
    for (int i = 0; i < mNumProcs; i++)
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
    for (int i = 0; i < mNumProcs; i++)
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