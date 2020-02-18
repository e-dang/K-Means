#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class IDistanceSumCalculator
{
public:
    virtual precision calcDistances(const Matrix<precision, int_size>* const data,
                                    const std::vector<precision>* const point,
                                    std::vector<precision>* const sqDistances,
                                    std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialDistanceSumCalculator : public IDistanceSumCalculator<precision, int_size>
{
public:
    precision calcDistances(const Matrix<precision, int_size>* const data, const std::vector<precision>* const point,
                            std::vector<precision>* const sqDistances,
                            std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override
    {
        precision distanceSum = 0.0;
        for (int_size i = 0; i < data->size(); i++)
        {
            sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), point->data(), point->size()), 2);
            distanceSum += sqDistances->at(i);
        }

        return distanceSum;
    }
};

template <typename precision = double, typename int_size = int32_t>
class OMPDistanceSumCalculator : public IDistanceSumCalculator<precision, int_size>
{
public:
    precision calcDistances(const Matrix<precision, int_size>* const data, const std::vector<precision>* const point,
                            std::vector<precision>* const sqDistances,
                            std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override
    {
        precision distanceSum = 0.0;
#pragma omp parallel for shared(distanceFunc), schedule(static), reduction(+ : distanceSum)
        for (int_size i = 0; i < data->size(); i++)
        {
            sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), point->data(), point->size()), 2);
            distanceSum += sqDistances->at(i);
        }

        return distanceSum;
    }
};
}  // namespace HPKmeans