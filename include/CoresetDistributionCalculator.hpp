#pragma once

#include <algorithm>

#include "DataClasses.hpp"
#include "Definitions.hpp"

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
                          std::vector<value_t>* const distribution) override
    {
        value_t partialQ = 0.5 * (1.0 / sqDistances->size());
        std::transform(
          sqDistances->begin(), sqDistances->end(), distribution->begin(),
          [&partialQ, &distanceSum](const value_t& dist) { return partialQ + (0.5 * dist / distanceSum); });
    }
};

class OMPCoresetDistributionCalculator : public ICoresetDistributionCalculator
{
public:
    void calcDistribution(const std::vector<value_t>* const sqDistances, const value_t& distanceSum,
                          std::vector<value_t>* const distribution) override
    {
        value_t partialQ = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
#pragma omp parallel for shared(sqDistances, distanceSum, distribution, partialQ), schedule(static)
        for (size_t i = 0; i < sqDistances->size(); i++)
        {
            distribution->at(i) = partialQ + 0.5 * sqDistances->at(i) / distanceSum;
        }
    }
};