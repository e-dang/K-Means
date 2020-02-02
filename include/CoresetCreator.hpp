#pragma once

#include <algorithm>
#include <memory>

#include "Averager.hpp"
#include "RandomSelector.hpp"
#include "mpi.h"

class AbstractCoresetCreator
{
protected:
    std::unique_ptr<IMultiWeightedRandomSelector> pSelector;
    std::unique_ptr<AbstractAverager> pAverager;

public:
    AbstractCoresetCreator(IMultiWeightedRandomSelector* selector, AbstractAverager* averager) :
        pSelector(selector), pAverager(averager)
    {
    }

    virtual ~AbstractCoresetCreator() {}

    virtual void createCoreset(Matrix* data, const int& sampleSize, Coreset* coreset, IDistanceFunctor* distanceFunc);

    virtual void finishClustering(Matrix* data, ClusterResults* clusterResults, IDistanceFunctor* distanceFunc);

    virtual void calcMean(Matrix* data, std::vector<value_t>* mean);

    virtual value_t calcDistsFromMean(Matrix* data, std::vector<value_t>* mean, std::vector<value_t>* sqDistances,
                                      IDistanceFunctor* distanceFunc);

    virtual void calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                  std::vector<value_t>* distribution);

    virtual void sampleDistribution(Matrix* data, std::vector<value_t>* distribution, const int& sampleSize,
                                    Coreset* coreset);
};

class SerialCoresetCreator : public AbstractCoresetCreator
{
public:
    SerialCoresetCreator(IMultiWeightedRandomSelector* selector, AbstractAverager* averager) :
        AbstractCoresetCreator(selector, averager)
    {
    }

    virtual ~SerialCoresetCreator() {}
};

class OMPCoresetCreator : public AbstractCoresetCreator
{
public:
    OMPCoresetCreator(IMultiWeightedRandomSelector* selector, AbstractAverager* averager) :
        AbstractCoresetCreator(selector, averager)
    {
    }

    virtual ~OMPCoresetCreator() {}

    value_t calcDistsFromMean(Matrix* data, std::vector<value_t>* mean, std::vector<value_t>* sqDistances,
                              IDistanceFunctor* distanceFunc) override;

    void calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* distribution) override;
};

class MPICoresetCreator : public AbstractCoresetCreator
{
private:
    Matrix chunkMeans;
    std::vector<int> mDataPerProc;
    std::vector<int> mDisplacements;
    std::vector<value_t> mDistanceSums;
    int mRank;
    int mNumProcs;
    int mSampleSize;
    int mTotalNumData;
    int mNumUniformSamples;
    int mNumNonUniformSamples;
    value_t mTotalDistanceSum;

public:
    MPICoresetCreator(const int& sampleSize, const int& totalNumData, IMultiWeightedRandomSelector* selector,
                      AbstractAverager* averager) :
        mSampleSize(sampleSize), mTotalNumData(totalNumData), AbstractCoresetCreator(selector, averager)
    {
        // auto mpiChunks = getMPIChunks(totalNumData);
        // mDataPerProc   = mpiChunks.dataPerProc;
        // mRank          = mpiChunk.rank;
        // mNumProcs      = mpiChunks.numProcs;

        MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mNumProcs);

        // number of datapoints allocated for each process to compute
        int chunk = mTotalNumData / mNumProcs;
        int scrap = chunk + (mTotalNumData % mNumProcs);
        for (int i = 0; i < mNumProcs; i++)
        {
            mDataPerProc.push_back(chunk);
            mDisplacements.push_back(i * chunk);
        }
        mDataPerProc.at(mNumProcs - 1) = scrap;
    }

    virtual ~MPICoresetCreator() {}

    void calcMean(Matrix* data, std::vector<value_t>* mean) override;

    value_t calcDistsFromMean(Matrix* data, std::vector<value_t>* mean, std::vector<value_t>* sqDistances,
                              IDistanceFunctor* distanceFunc) override;

    void calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* distribution) override;

    void sampleDistribution(Matrix* data, std::vector<value_t>* distribution, const int& sampleSize,
                            Coreset* coreset) override;

    void finishClustering(Matrix* data, ClusterResults* clusterResults, IDistanceFunctor* distanceFunc) override;
};