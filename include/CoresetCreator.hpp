#pragma once

#include <algorithm>
#include <memory>

#include "Averager.hpp"
#include "RandomSelector.hpp"

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