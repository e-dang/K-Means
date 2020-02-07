#pragma once

#include <memory>

#include "Averager.hpp"
#include "CoresetDistributionCalculator.hpp"
#include "DistanceSumCalculator.hpp"
#include "RandomSelector.hpp"
#include "Utils.hpp"

class AbstractCoresetCreator
{
protected:
    size_t mSampleSize;
    std::unique_ptr<IMultiWeightedRandomSelector> pSelector;
    std::unique_ptr<AbstractAverager> pAverager;
    std::unique_ptr<IDistanceSumCalculator> pDistSumCalc;
    std::shared_ptr<IDistanceFunctor> pDistanceFunc;

public:
    AbstractCoresetCreator(const size_t& sampleSize, IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                           IDistanceSumCalculator* distSumCalc, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        mSampleSize(sampleSize),
        pSelector(selector),
        pAverager(averager),
        pDistSumCalc(distSumCalc),
        pDistanceFunc(distanceFunc){};

    AbstractCoresetCreator(IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                           IDistanceSumCalculator* distSumCalc, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractCoresetCreator(NULL, selector, averager, distSumCalc, distanceFunc){};

    virtual ~AbstractCoresetCreator(){};

    void createCoreset(const Matrix* const data, Coreset* const coreset);

protected:
    virtual void calcMean(const Matrix* const data, std::vector<value_t>* const mean) = 0;

    virtual value_t calcDistsFromMean(const Matrix* const data, const std::vector<value_t>* const mean,
                                      std::vector<value_t>* const sqDistances) = 0;

    virtual void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                                  std::vector<value_t>* const distribution) = 0;

    virtual void sampleDistribution(const Matrix* const data, const std::vector<value_t>* const distribution,
                                    Coreset* const coreset) = 0;
};

class SharedMemoryCoresetCreator : public AbstractCoresetCreator
{
protected:
    std::unique_ptr<ICoresetDistributionCalculator> pDistrCalc;

public:
    SharedMemoryCoresetCreator(const size_t& sampleSize, IMultiWeightedRandomSelector* selector,
                               AbstractAverager* averager, IDistanceSumCalculator* distSumCalc,
                               ICoresetDistributionCalculator* distrCalc,
                               std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractCoresetCreator(sampleSize, selector, averager, distSumCalc, distanceFunc), pDistrCalc(distrCalc){};

    SharedMemoryCoresetCreator(IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                               IDistanceSumCalculator* distSumCalc, ICoresetDistributionCalculator* distrCalc,
                               std::shared_ptr<IDistanceFunctor> distanceFunc) :
        SharedMemoryCoresetCreator(NULL, selector, averager, distSumCalc, distrCalc, distanceFunc){};

    ~SharedMemoryCoresetCreator(){};

protected:
    void calcMean(const Matrix* const data, std::vector<value_t>* const mean) override;

    value_t calcDistsFromMean(const Matrix* const data, const std::vector<value_t>* const mean,
                              std::vector<value_t>* const sqDistances) override;

    void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* const distribution) override;

    void sampleDistribution(const Matrix* const data, const std::vector<value_t>* const distribution,
                            Coreset* const coreset) override;
};

class MPICoresetCreator : public AbstractCoresetCreator
{
protected:
    Matrix chunkMeans;
    std::vector<int> mLengths;
    std::vector<int> mDisplacements;
    std::vector<value_t> mDistanceSums;
    int mRank;
    int mNumProcs;
    size_t mTotalNumData;
    int_fast32_t mNumUniformSamples;
    int_fast32_t mNumNonUniformSamples;
    value_t mTotalDistanceSum;

public:
    MPICoresetCreator(const size_t& totalNumData, const size_t& sampleSize, IMultiWeightedRandomSelector* selector,
                      AbstractAverager* averager, IDistanceSumCalculator* distSumCalc,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractCoresetCreator(sampleSize, selector, averager, distSumCalc, distanceFunc)
    {
        setTotalNumData(totalNumData);
    }

    MPICoresetCreator(const size_t& sampleSize, IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                      IDistanceSumCalculator* distSumCalc, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        MPICoresetCreator(NULL, sampleSize, selector, averager, distSumCalc, distanceFunc){};

    ~MPICoresetCreator() {}

protected:
    void setTotalNumData(const size_t& totalNumData)
    {
        auto mpiData   = getMPIData(totalNumData);
        mRank          = mpiData.rank;
        mNumProcs      = mpiData.numProcs;
        mLengths       = mpiData.lengths;
        mDisplacements = mpiData.displacements;
        mTotalNumData  = totalNumData;
    }

    void calcMean(const Matrix* const data, std::vector<value_t>* const mean) override;

    value_t calcDistsFromMean(const Matrix* const data, const std::vector<value_t>* const mean,
                              std::vector<value_t>* const sqDistances) override;

    void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* const distribution) override;

    void sampleDistribution(const Matrix* const data, const std::vector<value_t>* const distribution,
                            Coreset* const coreset) override;

    void appendDataToCoreset(const Matrix* const data, Coreset* const coreset,
                             const std::vector<value_t>* const weights, const std::vector<value_t>* const distribution,
                             const int_fast32_t& numSamples);

    void calculateSamplingStrategy(std::vector<int_fast32_t>* const uniformSampleCounts,
                                   std::vector<int_fast32_t>* const nonUniformSampleCounts,
                                   const value_t& totalDistanceSums);

    void updateUniformSampleCounts(std::vector<int_fast32_t>* const uniformSampleCounts);

    void updateNonUniformSampleCounts(std::vector<int_fast32_t>* const nonUniformSampleCounts,
                                      const value_t& totalDistanceSums);

    void distributeCoreset(Coreset* const coreset);
};