#include "Strategies/DistanceSumCalculator.hpp"

value_t SerialDistanceSumCalculator::calcDistances(const Matrix* const data, const std::vector<value_t>* const point,
                                                   std::vector<value_t>* const sqDistances,
                                                   std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    value_t distanceSum = 0.0;
    for (int32_t i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), point->data(), point->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

value_t OMPDistanceSumCalculator::calcDistances(const Matrix* const data, const std::vector<value_t>* const point,
                                                std::vector<value_t>* const sqDistances,
                                                std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    value_t distanceSum = 0.0;
#pragma omp parallel for shared(distanceFunc), schedule(static), reduction(+ : distanceSum)
    for (int32_t i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), point->data(), point->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}