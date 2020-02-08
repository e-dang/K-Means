#include "Strategies/Averager.hpp"

#include <algorithm>

#include "omp.h"

#pragma omp declare reduction(+ : Matrix : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<value_t>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size(), omp_orig.getMaxNumData(), omp_orig.getNumFeatures()))

#pragma omp declare reduction(+ : std::vector<value_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<value_t>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void AbstractWeightedAverager::calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                                                const std::vector<int_fast32_t>* const dataAssignments,
                                                const std::vector<value_t>* const weights,
                                                const std::vector<value_t>* const weightSums,
                                                const int_fast32_t displacement)
{
    calculateSum(data, avgContainer, dataAssignments, weights, displacement);
    normalizeSum(avgContainer, weightSums);
}

void SerialWeightedMultiVectorAverager::calculateSum(const Matrix* const data, Matrix* const avgContainer,
                                                     const std::vector<int_fast32_t>* const dataAssignments,
                                                     const std::vector<value_t>* const weights,
                                                     const int_fast32_t displacement)
{
    for (int_fast32_t i = 0; i < data->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < data->getNumFeatures(); j++)
        {
            avgContainer->at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
        }
    }
}

void SerialWeightedMultiVectorAverager::normalizeSum(Matrix* const avgContainer,
                                                     const std::vector<value_t>* const weightSums)
{
    for (int_fast32_t i = 0; i < avgContainer->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < avgContainer->getNumFeatures(); j++)
        {
            avgContainer->at(i, j) /= weightSums->at(i);
        }
    }
}

void OMPWeightedMultiVectorAverager::calculateSum(const Matrix* const data, Matrix* const avgContainer,
                                                  const std::vector<int_fast32_t>* const dataAssignments,
                                                  const std::vector<value_t>* const weights,
                                                  const int_fast32_t displacement)
{
    Matrix& refContainer = *avgContainer;

#pragma omp parallel for shared(data, dataAssignments, weights), schedule(static), collapse(2), reduction(+:refContainer)
    for (int_fast32_t i = 0; i < data->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < data->getNumFeatures(); j++)
        {
            refContainer.at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
        }
    }
}

void OMPWeightedMultiVectorAverager::normalizeSum(Matrix* const avgContainer,
                                                  const std::vector<value_t>* const weightSums)
{
#pragma omp parallel for shared(avgContainer, weightSums), schedule(static), collapse(2)
    for (int_fast32_t i = 0; i < avgContainer->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < avgContainer->getNumFeatures(); j++)
        {
            avgContainer->at(i, j) /= weightSums->at(i);
        }
    }
}

void AbstractAverager::calculateAverage(const Matrix* const data, std::vector<value_t>* const avgContainer)
{
    calculateSum(data, avgContainer);
    normalizeSum(avgContainer, data->getNumData());
}

void SerialVectorAverager::calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer)
{
    for (int_fast32_t i = 0; i < data->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < data->getNumFeatures(); j++)
        {
            avgContainer->at(j) += data->at(i, j);
        }
    }
}

void SerialVectorAverager::normalizeSum(std::vector<value_t>* const avgContainer, const int_fast32_t& numData)
{
    for (size_t i = 0; i < avgContainer->size(); i++)
    {
        avgContainer->at(i) /= numData;
    }
}

void OMPVectorAverager::calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer)
{
    std::vector<value_t>& refContainer = *avgContainer;

#pragma omp parallel for shared(data, avgContainer), schedule(static), collapse(2), reduction(+ : refContainer)
    for (int_fast32_t i = 0; i < data->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < data->getNumFeatures(); j++)
        {
            refContainer[j] += data->at(i, j);
        }
    }
}

void OMPVectorAverager::normalizeSum(std::vector<value_t>* const avgContainer, const int_fast32_t& numData)
{
#pragma omp parallel for shared(avgContainer, numData), schedule(static)
    for (size_t i = 0; i < avgContainer->size(); i++)
    {
        avgContainer->at(i) /= numData;
    }
}