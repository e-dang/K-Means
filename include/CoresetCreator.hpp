#pragma once

#include <memory>

#include "Averager.hpp"
#include "RandomSelector.hpp"
#include "Utils.hpp"

class AbstractCoresetCreator
{
protected:
    int mSampleSize;
    std::unique_ptr<IMultiWeightedRandomSelector> pSelector;
    std::unique_ptr<AbstractAverager> pAverager;
    std::shared_ptr<IDistanceFunctor> pDistanceFunc;

public:
    AbstractCoresetCreator(const int& sampleSize, IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                           IDistanceFunctor* distanceFunc) :
        mSampleSize(sampleSize), pSelector(selector), pAverager(averager), pDistanceFunc(distanceFunc)
    {
    }

    virtual ~AbstractCoresetCreator() {}

    virtual void createCoreset(Matrix* data, Coreset* coreset);

    virtual void finishClustering(Matrix* data, ClusterResults* clusterResults);

    virtual void calcMean(Matrix* data, std::vector<value_t>* mean);

    virtual value_t calcDistsFromMean(Matrix* data, std::vector<value_t>* mean, std::vector<value_t>* sqDistances);

    virtual void calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                  std::vector<value_t>* distribution);

    virtual void sampleDistribution(Matrix* data, std::vector<value_t>* distribution, Coreset* coreset);
};

class SerialCoresetCreator : public AbstractCoresetCreator
{
public:
    SerialCoresetCreator(const int& sampleSize, IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                         IDistanceFunctor* distanceFunc) :
        AbstractCoresetCreator(sampleSize, selector, averager, distanceFunc)
    {
    }

    virtual ~SerialCoresetCreator() {}
};

class OMPCoresetCreator : public AbstractCoresetCreator
{
public:
    OMPCoresetCreator(const int& sampleSize, IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                      IDistanceFunctor* distanceFunc) :
        AbstractCoresetCreator(sampleSize, selector, averager, distanceFunc)
    {
    }

    virtual ~OMPCoresetCreator() {}

    value_t calcDistsFromMean(Matrix* data, std::vector<value_t>* mean, std::vector<value_t>* sqDistances) override;

    void calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* distribution) override;
};

class MPICoresetCreator : public AbstractCoresetCreator
{
private:
    Matrix chunkMeans;
    std::vector<int> mLengths;
    std::vector<int> mDisplacements;
    std::vector<value_t> mDistanceSums;
    int mRank;
    int mNumProcs;
    int mTotalNumData;
    int mNumUniformSamples;
    int mNumNonUniformSamples;
    value_t mTotalDistanceSum;

public:
    MPICoresetCreator(const int& totalNumData, const int& sampleSize, IMultiWeightedRandomSelector* selector,
                      AbstractAverager* averager, IDistanceFunctor* distanceFunc) :
        mTotalNumData(totalNumData), AbstractCoresetCreator(sampleSize, selector, averager, distanceFunc)
    {
        auto mpiData   = getMPIData(totalNumData);
        mRank          = mpiData.rank;
        mNumProcs      = mpiData.numProcs;
        mLengths       = mpiData.lengths;
        mDisplacements = mpiData.displacements;
    }

    virtual ~MPICoresetCreator() {}

    void calcMean(Matrix* data, std::vector<value_t>* mean) override;

    value_t calcDistsFromMean(Matrix* data, std::vector<value_t>* mean, std::vector<value_t>* sqDistances) override;

    void calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* distribution) override;

    void sampleDistribution(Matrix* data, std::vector<value_t>* distribution, Coreset* coreset) override;

    void finishClustering(Matrix* data, ClusterResults* clusterResults) override;

    void calculateSamplingStrategy(std::vector<int>* uniformSampleCounts, std::vector<int>* nonUniformSampleCounts,
                                   const value_t& totalDistanceSums);
    void updateUniformSampleCounts(std::vector<int>* uniformSampleCounts);
    void updateNonUniformSampleCounts(std::vector<int>* nonUniformSampleCounts, const value_t& totalDistanceSums);

    void appendDataToCoreset(Matrix* data, Coreset* coreset, std::vector<value_t>* weights,
                             std::vector<value_t>* distribution, const int& numSamples);

    void distributeCoreset(Coreset* coreset);
};