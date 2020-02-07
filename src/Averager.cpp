#include "Averager.hpp"

#include "omp.h"

void SerialWeightedMultiVectorAverager::calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                                                         const std::vector<int_fast32_t>* const dataAssignments,
                                                         const std::vector<value_t>* const weights,
                                                         const std::vector<value_t>* const weightSums)
{
    calculateSum(data, avgContainer, dataAssignments, weights);
    normalizeSum(avgContainer, weightSums);
}

void SerialWeightedMultiVectorAverager::calculateSum(const Matrix* const data, Matrix* const avgContainer,
                                                     const std::vector<int_fast32_t>* const dataAssignments,
                                                     const std::vector<value_t>* const weights)
{
    for (int_fast32_t i = 0; i < data->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < data->getNumFeatures(); j++)
        {
            avgContainer->at(dataAssignments->at(i), j) += weights->at(i) * data->at(i, j);
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
    std::vector<value_t> copyContainer(avgContainer->size(), 0);
    auto copyContainerData = copyContainer.data();

#pragma omp parallel for shared(data, avgContainer), schedule(static), collapse(2), reduction(+ : copyContainerData[:data->getNumFeatures()])
    for (int_fast32_t i = 0; i < data->getNumData(); i++)
    {
        for (int_fast32_t j = 0; j < data->getNumFeatures(); j++)
        {
            copyContainer[j] += data->at(i, j);
        }
    }

    std::copy(copyContainer.begin(), copyContainer.end(), avgContainer->begin());
}

void OMPVectorAverager::normalizeSum(std::vector<value_t>* const avgContainer, const int_fast32_t& numData)
{
#pragma omp parallel for shared(avgContainer, numData), schedule(static)
    for (size_t i = 0; i < avgContainer->size(); i++)
    {
        avgContainer->at(i) /= numData;
    }
}