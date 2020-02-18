#pragma once

#include <algorithm>

#include "Containers/DataClasses.hpp"
#include "omp.h"

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class AbstractWeightedAverager
{
public:
    void calculateAverage(const Matrix<precision, int_size>* const data,
                          Matrix<precision, int_size>* const avgContainer,
                          const std::vector<int_size>* const dataAssignments,
                          const std::vector<precision>* const weights, const std::vector<precision>* const weightSums,
                          const int_size displacement = 0)
    {
        calculateSum(data, avgContainer, dataAssignments, weights, displacement);
        normalizeSum(avgContainer, weightSums);
    }

    virtual void calculateSum(const Matrix<precision, int_size>* const data,
                              Matrix<precision, int_size>* const avgContainer,
                              const std::vector<int_size>* const dataAssignments,
                              const std::vector<precision>* const weights, const int_size displacement = 0) = 0;

    virtual void normalizeSum(Matrix<precision, int_size>* const avgContainer,
                              const std::vector<precision>* const weightSums) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialWeightedMultiVectorAverager : public AbstractWeightedAverager<precision, int_size>
{
public:
    void calculateSum(const Matrix<precision, int_size>* const data, Matrix<precision, int_size>* const avgContainer,
                      const std::vector<int_size>* const dataAssignments, const std::vector<precision>* const weights,
                      const int_size displacement = 0) override
    {
        for (int32_t i = 0; i < data->size(); i++)
        {
            for (int32_t j = 0; j < data->cols(); j++)
            {
                avgContainer->at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
            }
        }
    }

    void normalizeSum(Matrix<precision, int_size>* const avgContainer,
                      const std::vector<precision>* const weightSums) override
    {
        for (int32_t i = 0; i < avgContainer->size(); i++)
        {
            for (int32_t j = 0; j < avgContainer->cols(); j++)
            {
                avgContainer->at(i, j) /= weightSums->at(i);
            }
        }
    }
};

template <typename precision = double, typename int_size = int32_t>
class OMPWeightedMultiVectorAverager : public AbstractWeightedAverager<precision, int_size>
{
public:
    void calculateSum(const Matrix<precision, int_size>* const data, Matrix<precision, int_size>* const avgContainer,
                      const std::vector<int_size>* const dataAssignments, const std::vector<precision>* const weights,
                      const int_size displacement = 0) override
    {
        for (int32_t i = 0; i < data->size(); i++)
        {
            for (int32_t j = 0; j < data->cols(); j++)
            {
                avgContainer->at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
            }
        }
        // #pragma omp declare reduction(+ : Matrix<precision, int_size> : \
//                               std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<precision>())) \
//                     initializer(omp_priv = decltype(omp_orig)(omp_orig.rows(), omp_orig.cols()))

        //         Matrix<precision, int_size>& refContainer = *avgContainer;

        // #pragma omp parallel for schedule(static), collapse(2), reduction(+ : refContainer)
        //         for (int32_t i = 0; i < data->size(); i++)
        //         {
        //             for (int32_t j = 0; j < data->cols(); j++)
        //             {
        //                 refContainer.at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
        //             }
        //         }
    }

    void normalizeSum(Matrix<precision, int_size>* const avgContainer,
                      const std::vector<precision>* const weightSums) override
    {
#pragma omp parallel for schedule(static), collapse(2)
        for (int32_t i = 0; i < avgContainer->size(); i++)
        {
            for (int32_t j = 0; j < avgContainer->cols(); j++)
            {
                avgContainer->at(i, j) /= weightSums->at(i);
            }
        }
    }
};

template <typename precision = double, typename int_size = int32_t>
class AbstractAverager
{
public:
    void calculateAverage(const Matrix<precision, int_size>* const data, std::vector<precision>* const avgContainer)
    {
        calculateSum(data, avgContainer);
        normalizeSum(avgContainer, data->size());
    }

    virtual void calculateSum(const Matrix<precision, int_size>* const data,
                              std::vector<precision>* const avgContainer) = 0;

    virtual void normalizeSum(std::vector<precision>* const avgContainer, const int_size& numData) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialVectorAverager : public AbstractAverager<precision, int_size>
{
public:
    void calculateSum(const Matrix<precision, int_size>* const data,
                      std::vector<precision>* const avgContainer) override
    {
        for (int32_t i = 0; i < data->size(); i++)
        {
            for (int32_t j = 0; j < data->cols(); j++)
            {
                avgContainer->at(j) += data->at(i, j);
            }
        }
    }

    void normalizeSum(std::vector<precision>* const avgContainer, const int_size& numData) override
    {
        for (size_t i = 0; i < avgContainer->size(); i++)
        {
            avgContainer->at(i) /= numData;
        }
    }
};

template <typename precision = double, typename int_size = int32_t>
class OMPVectorAverager : public AbstractAverager<precision, int_size>
{
public:
    void calculateSum(const Matrix<precision, int_size>* const data,
                      std::vector<precision>* const avgContainer) override
    {
        for (int32_t i = 0; i < data->size(); i++)
        {
            for (int32_t j = 0; j < data->cols(); j++)
            {
                avgContainer->at(j) += data->at(i, j);
            }
        }
        // #pragma omp declare reduction(+ : std::vector<precision> : \
//                               std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<precision>())) \
//                                 initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

        //         std::vector<precision>& refContainer = *avgContainer;

        // #pragma omp parallel for schedule(static), collapse(2), reduction(+ : refContainer)
        //         for (int32_t i = 0; i < data->size(); i++)
        //         {
        //             for (int32_t j = 0; j < data->cols(); j++)
        //             {
        //                 refContainer[j] += data->at(i, j);
        //             }
        //         }
    }

    void normalizeSum(std::vector<precision>* const avgContainer, const int_size& numData) override
    {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < avgContainer->size(); i++)
        {
            avgContainer->at(i) /= numData;
        }
    }
};
}  // namespace HPKmeans