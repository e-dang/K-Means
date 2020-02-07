#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class ICoresetDistributionCalculator
{
public:
    virtual void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                                  std::vector<value_t>* const distribution) = 0;
};

class SerialCoresetDistributionCalculator : public ICoresetDistributionCalculator
{
public:
    void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* const distribution) override;
};

class OMPCoresetDistributionCalculator : public ICoresetDistributionCalculator
{
public:
    void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* const distribution) override;
};