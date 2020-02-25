#pragma once

#include <hpkmeans/data_types/kmeans_state.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
class IDistanceSumCalculator
{
public:
    virtual ~IDistanceSumCalculator() = default;

    virtual precision calcDistances(KmeansState<precision, int_size>* kmeansState,
                                    const std::vector<precision>* const point,
                                    std::vector<precision>* const sqDistances) = 0;
};

template <typename precision, typename int_size>
class SerialDistanceSumCalculator : public IDistanceSumCalculator<precision, int_size>
{
public:
    ~SerialDistanceSumCalculator() = default;

    precision calcDistances(KmeansState<precision, int_size>* kmeansState, const std::vector<precision>* const point,
                            std::vector<precision>* const sqDistances) override;
};

template <typename precision, typename int_size>
class OMPDistanceSumCalculator : public IDistanceSumCalculator<precision, int_size>
{
public:
    ~OMPDistanceSumCalculator() = default;

    precision calcDistances(KmeansState<precision, int_size>* kmeansState, const std::vector<precision>* const point,
                            std::vector<precision>* const sqDistances) override;
};

template <typename precision, typename int_size>
precision SerialDistanceSumCalculator<precision, int_size>::calcDistances(KmeansState<precision, int_size>* kmeansState,
                                                                          const std::vector<precision>* const point,
                                                                          std::vector<precision>* const sqDistances)
{
    precision distanceSum = 0.0;
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        sqDistances->at(i) = std::pow((*kmeansState)(kmeansState->dataAt(i), point->data()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

template <typename precision, typename int_size>
precision OMPDistanceSumCalculator<precision, int_size>::calcDistances(KmeansState<precision, int_size>* kmeansState,
                                                                       const std::vector<precision>* const point,
                                                                       std::vector<precision>* const sqDistances)
{
    precision distanceSum = 0.0;
#pragma omp parallel for schedule(static), reduction(+ : distanceSum)
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        sqDistances->at(i) = std::pow((*kmeansState)(kmeansState->dataAt(i), point->data()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}
}  // namespace HPKmeans