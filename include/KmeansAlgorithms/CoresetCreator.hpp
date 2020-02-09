#pragma once

#include <memory>

#include "Strategies/Averager.hpp"
#include "Strategies/CoresetDistributionCalculator.hpp"
#include "Strategies/DistanceSumCalculator.hpp"
#include "Strategies/RandomSelector.hpp"
#include "Utils/Utils.hpp"

class AbstractCoresetCreator
{
protected:
    int32_t mSampleSize;
    std::unique_ptr<IMultiWeightedRandomSelector> pSelector;
    std::unique_ptr<AbstractAverager> pAverager;
    std::unique_ptr<IDistanceSumCalculator> pDistSumCalc;
    std::shared_ptr<IDistanceFunctor> pDistanceFunc;

public:
    AbstractCoresetCreator(const int32_t& sampleSize, IMultiWeightedRandomSelector* selector,
                           AbstractAverager* averager, IDistanceSumCalculator* distSumCalc,
                           std::shared_ptr<IDistanceFunctor> distanceFunc) :
        mSampleSize(sampleSize),
        pSelector(selector),
        pAverager(averager),
        pDistSumCalc(distSumCalc),
        pDistanceFunc(distanceFunc){};

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
    SharedMemoryCoresetCreator(const int32_t& sampleSize, IMultiWeightedRandomSelector* selector,
                               AbstractAverager* averager, IDistanceSumCalculator* distSumCalc,
                               ICoresetDistributionCalculator* distrCalc,
                               std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractCoresetCreator(sampleSize, selector, averager, distSumCalc, distanceFunc), pDistrCalc(distrCalc){};

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
    std::vector<int32_t> mLengths;
    std::vector<int32_t> mDisplacements;
    std::vector<value_t> mDistanceSums;
    int mRank;
    int mNumProcs;
    int32_t mTotalNumData;
    int32_t mNumUniformSamples;
    int32_t mNumNonUniformSamples;
    value_t mTotalDistanceSum;

public:
    MPICoresetCreator(const int32_t& sampleSize, IMultiWeightedRandomSelector* selector, AbstractAverager* averager,
                      IDistanceSumCalculator* distSumCalc, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractCoresetCreator(sampleSize, selector, averager, distSumCalc, distanceFunc){};

    ~MPICoresetCreator() {}

protected:
    void setTotalNumData(const int32_t& totalNumData)
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
                             const int32_t& numSamples);

    void calculateSamplingStrategy(std::vector<int32_t>* const uniformSampleCounts,
                                   std::vector<int32_t>* const nonUniformSampleCounts,
                                   const value_t& totalDistanceSums);

    void updateUniformSampleCounts(std::vector<int32_t>* const uniformSampleCounts);

    void updateNonUniformSampleCounts(std::vector<int32_t>* const nonUniformSampleCounts,
                                      const value_t& totalDistanceSums);

    void distributeCoreset(Coreset* const coreset);
};