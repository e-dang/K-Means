#include "Averager.hpp"

#include "omp.h"

void WeightedMultiVectorAverager::calculateSum(Matrix* data, Matrix* avgContainer, std::vector<int>* dataAssignments,
                                               std::vector<value_t>* weights)
{
    for (int i = 0; i < data->getNumData(); i++)
    {
        for (int j = 0; j < data->getNumFeatures(); j++)
        {
            avgContainer->at(dataAssignments->at(i), j) += weights->at(i) * data->at(i, j);
        }
    }
}

void WeightedMultiVectorAverager::normalizeSum(Matrix* avgContainer, std::vector<value_t>* weightSums)
{
    for (int i = 0; i < avgContainer->getNumData(); i++)
    {
        for (int j = 0; j < avgContainer->getNumFeatures(); j++)
        {
            avgContainer->at(i, j) /= weightSums->at(i);
        }
    }
}

void WeightedMultiVectorAverager::calculateAverage(Matrix* data, Matrix* avgContainer,
                                                   std::vector<int>* dataAssignments, std::vector<value_t>* weights,
                                                   std::vector<value_t>* weightSums)
{
    calculateSum(data, avgContainer, dataAssignments, weights);
    normalizeSum(avgContainer, weightSums);
}

void OMPWeightedMultiVectorAverager::normalizeSum(Matrix* avgContainer, std::vector<value_t>* weightSums)
{
#pragma omp parallel for shared(avgContainer, weightSums), schedule(static), collapse(2)
    for (int i = 0; i < avgContainer->getNumData(); i++)
    {
        for (int j = 0; j < avgContainer->getNumFeatures(); j++)
        {
            avgContainer->at(i, j) /= weightSums->at(i);
        }
    }
}

void VectorAverager::calculateSum(Matrix* data, std::vector<value_t>* avgContainer)
{
    for (int i = 0; i < data->getNumData(); i++)
    {
        for (int j = 0; j < data->getNumFeatures(); j++)
        {
            avgContainer->at(j) += data->at(i, j);
        }
    }
}

void VectorAverager::normalizeSum(std::vector<value_t>* avgContainer, int numData)
{
    for (int i = 0; i < avgContainer->size(); i++)
    {
        avgContainer->at(i) /= numData;
    }
}

void VectorAverager::calculateAverage(Matrix* data, std::vector<value_t>* avgContainer)
{
    calculateSum(data, avgContainer);
    normalizeSum(avgContainer, data->getNumData());
}

void OMPVectorAverager::calculateSum(Matrix* data, std::vector<value_t>* avgContainer)
{
    std::vector<value_t> copyContainer(avgContainer->size(), 0);
    auto copyContainerData = copyContainer.data();

#pragma omp parallel for shared(data, avgContainer), schedule(static), collapse(2), reduction(+ : copyContainerData[:data->getNumFeatures()])
    for (int i = 0; i < data->getNumData(); i++)
    {
        for (int j = 0; j < data->getNumFeatures(); j++)
        {
            copyContainer[j] += data->at(i, j);
        }
    }

    std::copy(copyContainer.begin(), copyContainer.end(), avgContainer->begin());
}

void OMPVectorAverager::normalizeSum(std::vector<value_t>* avgContainer, int numData)
{
#pragma omp parallel for shared(avgContainer, numData), schedule(static)
    for (int i = 0; i < avgContainer->size(); i++)
    {
        avgContainer->at(i) /= numData;
    }
}

void OMPVectorAverager::calculateAverage(Matrix* data, std::vector<value_t>* avgContainer)
{
    calculateSum(data, avgContainer);
    normalizeSum(avgContainer, data->getNumData());
}