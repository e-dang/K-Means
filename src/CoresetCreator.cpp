#include "CoresetCreator.hpp"

#include <iostream>
#include <numeric>

#include "Utils.hpp"

// #include "boost/generator_iterator.hpp"
// #include "boost/random.hpp"
#include "mpi.h"
#include "omp.h"

// typedef boost::mt19937 RNGType;

void AbstractCoresetCreator::createCoreset(Matrix* data, const int& sampleSize, Coreset* coreset,
                                           IDistanceFunctor* distanceFunc)
{
    std::vector<value_t> mean(data->getNumFeatures());
    std::vector<value_t> sqDistances(data->getNumData());
    std::vector<value_t> distribution(data->getNumData(), 0);

    calcMean(data, &mean);

    value_t distanceSum = calcDistsFromMean(data, &mean, &sqDistances, distanceFunc);

    calcDistribution(&sqDistances, distanceSum, &distribution);

    sampleDistribution(data, &distribution, sampleSize, coreset);
    // sleep(5);
}

void AbstractCoresetCreator::calcMean(Matrix* data, std::vector<value_t>* mean)
{
    pAverager->calculateAverage(data, mean);
}

value_t AbstractCoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                                  std::vector<value_t>* sqDistances, IDistanceFunctor* distanceFunc)
{
    value_t distanceSum = 0;
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

void AbstractCoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                              std::vector<value_t>* distribution)
{
    value_t partOne = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
    for (int i = 0; i < sqDistances->size(); i++)
    {
        distribution->at(i) = partOne + 0.5 * sqDistances->at(i) / distanceSum;
    }
}

void AbstractCoresetCreator::sampleDistribution(Matrix* data, std::vector<value_t>* distribution, const int& sampleSize,
                                                Coreset* coreset)
{
    auto selectedIdxs = pSelector->select(distribution, sampleSize);
    for (auto& idx : selectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (sampleSize * distribution->at(idx)));
    }
}

value_t OMPCoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                             std::vector<value_t>* sqDistances, IDistanceFunctor* distanceFunc)
{
    value_t distanceSum = 0;
#pragma omp parallel for shared(data, mean, sqDistances, distanceFunc), schedule(static), reduction(+ : distanceSum)
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

void OMPCoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                         std::vector<value_t>* distribution)
{
    value_t partOne = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
#pragma omp parallel for shared(sqDistances, distanceSum, distribution, partOne), schedule(static)
    for (int i = 0; i < sqDistances->size(); i++)
    {
        distribution->at(i) = partOne + 0.5 * sqDistances->at(i) / distanceSum;
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
        int numData = std::accumulate(mDataPerProc.begin(), mDataPerProc.end(), 0);
        pAverager->calculateSum(&chunkMeans, mean);
        pAverager->normalizeSum(mean, numData);
    }

    MPI_Bcast(mean->data(), mean->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

value_t MPICoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                             std::vector<value_t>* sqDistances, IDistanceFunctor* distanceFunc)
{
    // calculate local quantization errors
    value_t localDistanceSum = 0;
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) += std::pow((*distanceFunc)(data->at(i), mean->data(), mean->size()), 2);
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
        for (int i = 0; i < mSampleSize; i++)
        {
            value_t randNum = getRandFloat01MPI();
            if (randNum > 0.5)
            {
                randNum           = getRandFloat01MPI() * mTotalNumData;
                int cumulativeSum = 0;
                for (int j = 0; j < mNumProcs; j++)
                {
                    cumulativeSum += mDataPerProc[j];
                    if (cumulativeSum >= randNum)
                    {
                        uniformSampleCounts.at(j)++;
                        break;
                    }
                }
            }
            else
            {
                randNum               = getRandFloat01MPI() * totalDistanceSums;
                value_t cumulativeSum = 0;
                for (int j = 0; j < mNumProcs; j++)
                {
                    cumulativeSum += mDistanceSums.at(j);
                    if (cumulativeSum >= randNum)
                    {
                        nonUniformSampleCounts.at(j)++;
                        break;
                    }
                }
            }
        }
    }

    MPI_Bcast(&totalDistanceSums, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatter(uniformSampleCounts.data(), 1, MPI_INT, &mNumUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nonUniformSampleCounts.data(), 1, MPI_INT, &mNumNonUniformSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < distribution->size(); i++)
    {
        distribution->at(i) = sqDistances->at(i) / totalDistanceSums;
    }
}

void MPICoresetCreator::sampleDistribution(Matrix* data, std::vector<value_t>* distribution, const int& sampleSize,
                                           Coreset* coreset)
{
    RNGType rng(time(NULL));
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    std::vector<value_t> uniformWeights(distribution->size(), 1.0 / mTotalNumData);
    std::vector<int> uniformSelectedIdxs = pSelector->select(&uniformWeights, mNumUniformSamples);
    for (auto& idx : uniformSelectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (sampleSize * distribution->at(idx)));
    }

    std::vector<int> nonUniformSelectedIdxs = pSelector->select(distribution, mNumNonUniformSamples);
    for (auto& idx : nonUniformSelectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (sampleSize * distribution->at(idx)));
    }

    int numCoresetData = coreset->weights.size();
    std::vector<int> coresetNumDataPerProc(mNumProcs);
    MPI_Allgather(&numCoresetData, 1, MPI_INT, coresetNumDataPerProc.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> lengths(mNumProcs);
    std::vector<int> matrixDisplacements(mNumProcs, 0);
    std::vector<int> vectorDisplacements(mNumProcs, 0);
    for (int i = 0; i < mNumProcs; i++)
    {
        lengths.at(i) = coresetNumDataPerProc.at(i) * data->getNumFeatures();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + lengths.at(i - 1);
            vectorDisplacements.at(i) = vectorDisplacements.at(i - 1) + coresetNumDataPerProc.at(i - 1);
        }
    }
    Coreset fullCoreset{ Matrix(mSampleSize, data->getNumFeatures()), std::vector<value_t>(mSampleSize) };
    fullCoreset.data.resize(mSampleSize);

    MPI_Gatherv(coreset->data.data(), coreset->data.size(), MPI_FLOAT, fullCoreset.data.data(), lengths.data(),
                matrixDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(coreset->weights.data(), coreset->weights.size(), MPI_FLOAT, fullCoreset.weights.data(),
                coresetNumDataPerProc.data(), vectorDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    int chunk = mSampleSize / mNumProcs;
    int scrap = chunk + (mSampleSize % mNumProcs);
    for (int i = 0; i < mNumProcs; i++)
    {
        lengths.at(i)             = chunk;
        vectorDisplacements.at(i) = i * chunk;
    }
    lengths.at(mNumProcs - 1) = scrap;

    coreset->data.resize(lengths.at(mRank));
    coreset->weights.resize(lengths.at(mRank));

    MPI_Scatterv(fullCoreset.weights.data(), lengths.data(), vectorDisplacements.data(), MPI_FLOAT,
                 coreset->weights.data(), coreset->weights.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < mNumProcs; i++)
    {
        lengths.at(i) *= data->getNumFeatures();
        if (i != 0)
        {
            matrixDisplacements.at(i) = matrixDisplacements.at(i - 1) + lengths.at(i - 1);
        }
    }
    MPI_Scatterv(fullCoreset.data.data(), lengths.data(), matrixDisplacements.data(), MPI_FLOAT, coreset->data.data(),
                 lengths.at(mRank), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void AbstractCoresetCreator::finishClustering(Matrix* data, ClusterResults* clusterResults,
                                              IDistanceFunctor* distanceFunc)
{
    for (int i = 0; i < data->getNumData(); i++)
    {
        // auto closestCluster =
        //   pFinder->findClosestCluster(data->at(i), clusterResults->mClusterData.mClusters, pDistanceFunc);

        int clusterIdx;
        value_t minDistance = -1;
        auto& clusters      = clusterResults->mClusterData.mClusters;
        for (int j = 0; j < clusters.getNumData(); j++)
        {
            value_t tempDistance = (*distanceFunc)(data->at(i), clusters.at(j), clusters.getNumFeatures());
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

void MPICoresetCreator::finishClustering(Matrix* data, ClusterResults* clusterResults, IDistanceFunctor* distanceFunc)
{
    for (int i = 0; i < data->getNumData(); i++)
    {
        // auto closestCluster =
        //   pFinder->findClosestCluster(data->at(i), clusterResults->mClusterData.mClusters, pDistanceFunc);

        int clusterIdx;
        value_t minDistance = -1;
        auto& clusters      = clusterResults->mClusterData.mClusters;
        for (int j = 0; j < clusters.getNumData(); j++)
        {
            value_t tempDistance = (*distanceFunc)(data->at(i), clusters.at(j), clusters.getNumFeatures());
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
    MPI_Allgatherv(MPI_IN_PLACE, mDataPerProc.at(mRank), MPI_INT, clusterResults->mClusterData.mClustering.data(),
                   mDataPerProc.data(), mDisplacements.data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, mDataPerProc.at(mRank), MPI_FLOAT, clusterResults->mSqDistances.data(),
                   mDataPerProc.data(), mDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &clusterResults->mError, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}