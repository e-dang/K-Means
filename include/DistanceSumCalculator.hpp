#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"

class IDistanceSumCalculator
{
public:
    virtual value_t calcDistances(const Matrix* const data, const std::vector<value_t>* const point,
                                  std::vector<value_t>* const sqDistances,
                                  std::shared_ptr<IDistanceFunctor> distanceFunc) = 0;
};

class SerialDistanceSumCalculator : public IDistanceSumCalculator
{
public:
    value_t calcDistances(const Matrix* const data, const std::vector<value_t>* const point,
                          std::vector<value_t>* const sqDistances,
                          std::shared_ptr<IDistanceFunctor> distanceFunc) override
    {
        value_t distanceSum = 0.0;
        for (int_fast32_t i = 0; i < data->getNumData(); i++)
        {
            sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), point->data(), point->size()), 2);
            distanceSum += sqDistances->at(i);
        }

        return distanceSum;
    }
};

class OMPDistanceSumCalculator : public IDistanceSumCalculator
{
public:
    value_t calcDistances(const Matrix* const data, const std::vector<value_t>* const point,
                          std::vector<value_t>* const sqDistances,
                          std::shared_ptr<IDistanceFunctor> distanceFunc) override
    {
        value_t distanceSum = 0.0;
#pragma omp parallel for shared(data, point, sqDistances, distanceFunc), schedule(static), reduction(+ : distanceSum)
        for (int_fast32_t i = 0; i < data->getNumData(); i++)
        {
            sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), point->data(), point->size()), 2);
            distanceSum += sqDistances->at(i);
        }

        return distanceSum;
    }
};
