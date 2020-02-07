#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

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
                          std::shared_ptr<IDistanceFunctor> distanceFunc) override;
};

class OMPDistanceSumCalculator : public IDistanceSumCalculator
{
public:
    value_t calcDistances(const Matrix* const data, const std::vector<value_t>* const point,
                          std::vector<value_t>* const sqDistances, std::shared_ptr<IDistanceFunctor> distanceFunc);
};
