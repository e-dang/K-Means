#include "CoresetCreator.hpp"

#include "omp.h"

void AbstractCoresetCreator::createCoreset(Matrix* data, const int& sampleSize, Coreset* coreset,
                                           IDistanceFunctor* distanceFunc)
{
    std::vector<value_t> mean(data->getNumData());
    std::vector<value_t> sqDistances(data->getNumData());
    std::vector<value_t> distribution(data->getNumData(), 0);

    pAverager->calculateAverage(data, &mean);

    value_t distanceSum = calcDistsFromMean(data, &mean, &sqDistances, distanceFunc);

    calcDistribution(&sqDistances, distanceSum, &distribution);

    sampleDistribution(data, &distribution, sampleSize, coreset);
}

value_t AbstractCoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                                  std::vector<value_t>* sqDistances, IDistanceFunctor* distanceFunc)
{
    value_t distanceSum = 0;
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

void AbstractCoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                              std::vector<value_t>* distribution)
{
    value_t partOne = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
    for (int i = 0; i < sqDistances->size(); i++)
    {
        distribution->at(i) = partOne + 0.5 * sqDistances->at(i) / distanceSum;
    }
}

void AbstractCoresetCreator::sampleDistribution(Matrix* data, std::vector<value_t>* distribution, const int& sampleSize,
                                                Coreset* coreset)
{
    auto selectedIdxs = pSelector->select(distribution, sampleSize);
    for (auto& idx : selectedIdxs)
    {
        coreset->data.appendDataPoint(data->at(idx));
        coreset->weights.push_back(1.0 / (sampleSize * distribution->at(idx)));
    }
}

value_t OMPCoresetCreator::calcDistsFromMean(Matrix* data, std::vector<value_t>* mean,
                                             std::vector<value_t>* sqDistances, IDistanceFunctor* distanceFunc)
{
    value_t distanceSum = 0;
#pragma omp parallel for shared(data, mean, sqDistances, distanceFunc), schedule(static), reduction(+ : distanceSum)
    for (int i = 0; i < data->getNumData(); i++)
    {
        sqDistances->at(i) = std::pow((*distanceFunc)(data->at(i), mean->data(), mean->size()), 2);
        distanceSum += sqDistances->at(i);
    }

    return distanceSum;
}

void OMPCoresetCreator::calcDistribution(std::vector<value_t>* sqDistances, const value_t& distanceSum,
                                         std::vector<value_t>* distribution)
{
    value_t partOne = 0.5 * (1.0 / sqDistances->size());  // portion of distribution calculation that is constant
#pragma omp parallel for shared(sqDistances, distanceSum, distribution, partOne), schedule(static)
    for (int i = 0; i < sqDistances->size(); i++)
    {
        distribution->at(i) = partOne + 0.5 * sqDistances->at(i) / distanceSum;
    }
}
