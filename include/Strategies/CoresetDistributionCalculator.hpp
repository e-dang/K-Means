#pragma once

#include <algorithm>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class ICoresetDistributionCalculator
{
public:
    virtual void calcDistribution(const std::vector<precision>* const sqDistances, const precision& distanceSum,
                                  std::vector<precision>* const distribution) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialCoresetDistributionCalculator : public ICoresetDistributionCalculator<precision, int_size>
{
public:
    void calcDistribution(const std::vector<precision>* const sqDistances, const precision& distanceSum,
                          std::vector<precision>* const distribution) override
    {
        precision partialQ = 0.5 * (1.0 / sqDistances->size());
        std::transform(
          sqDistances->begin(), sqDistances->end(), distribution->begin(),
          [&partialQ, &distanceSum](const precision& dist) { return partialQ + (0.5 * dist / distanceSum); });
    }
};

template <typename precision = double, typename int_size = int32_t>
class OMPCoresetDistributionCalculator : public ICoresetDistributionCalculator<precision, int_size>
{
public:
    void calcDistribution(const std::vector<precision>* const sqDistances, const precision& distanceSum,
                          std::vector<precision>* const distribution) override
    {
        precision partialQ = 0.5 * (1.0 / sqDistances->size());
#pragma omp parallel for shared(partialQ), schedule(static)
        for (size_t i = 0; i < sqDistances->size(); i++)
        {
            distribution->at(i) = partialQ + 0.5 * sqDistances->at(i) / distanceSum;
        }
    }
};
}  // namespace HPKmeans