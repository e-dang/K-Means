#pragma once

#include <algorithm>

#include "Containers/DataClasses.hpp"
namespace HPKmeans
{
template <typename precision, typename int_size>
class ICoresetDistributionCalculator
{
public:
    virtual ~ICoresetDistributionCalculator() = default;

    virtual void calcDistribution(const std::vector<precision>* const sqDistances, const precision& distanceSum,
                                  std::vector<precision>* const distribution) = 0;
};

template <typename precision, typename int_size>
class SerialCoresetDistributionCalculator : public ICoresetDistributionCalculator<precision, int_size>
{
public:
    ~SerialCoresetDistributionCalculator() = default;

    void calcDistribution(const std::vector<precision>* const sqDistances, const precision& distanceSum,
                          std::vector<precision>* const distribution) override;
};

template <typename precision, typename int_size>
class OMPCoresetDistributionCalculator : public ICoresetDistributionCalculator<precision, int_size>
{
public:
    ~OMPCoresetDistributionCalculator() = default;

    void calcDistribution(const std::vector<precision>* const sqDistances, const precision& distanceSum,
                          std::vector<precision>* const distribution) override;
};

template <typename precision, typename int_size>
void SerialCoresetDistributionCalculator<precision, int_size>::calcDistribution(
  const std::vector<precision>* const sqDistances, const precision& distanceSum,
  std::vector<precision>* const distribution)
{
    precision partialQ = 0.5 * (1.0 / sqDistances->size());
    std::transform(sqDistances->begin(), sqDistances->end(), distribution->begin(),
                   [&partialQ, &distanceSum](const precision& dist) { return partialQ + (0.5 * dist / distanceSum); });
}

template <typename precision, typename int_size>
void OMPCoresetDistributionCalculator<precision, int_size>::calcDistribution(
  const std::vector<precision>* const sqDistances, const precision& distanceSum,
  std::vector<precision>* const distribution)
{
    precision partialQ = 0.5 * (1.0 / sqDistances->size());
#pragma omp parallel for shared(partialQ), schedule(static)
    for (size_t i = 0; i < sqDistances->size(); ++i)
    {
        distribution->at(i) = partialQ + 0.5 * sqDistances->at(i) / distanceSum;
    }
}
}  // namespace HPKmeans