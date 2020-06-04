#pragma once

#include <omp.h>

#include <algorithm>
#include <hpkmeans/data_types/matrix.hpp>

namespace HPKmeans
{
#pragma omp declare reduction(+ : std::vector<float> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(+ : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#pragma omp declare reduction(+ : Matrix<float, int> : \
                               std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.rows(), omp_orig.cols(), true))

#pragma omp declare reduction(+ : Matrix<double, int> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                                initializer(omp_priv = decltype(omp_orig)(omp_orig.rows(), omp_orig.cols(), true))

template <typename precision, typename int_size>
class AbstractWeightedAverager
{
public:
    virtual ~AbstractWeightedAverager() = default;

    void calculateAverage(const Matrix<precision, int_size>* const data,
                          Matrix<precision, int_size>* const avgContainer,
                          const std::vector<int_size>* const dataAssignments,
                          const std::vector<precision>* const weights, const std::vector<precision>* const weightSums,
                          const int_size displacement = 0);

    virtual void calculateSum(const Matrix<precision, int_size>* const data,
                              Matrix<precision, int_size>* const avgContainer,
                              const std::vector<int_size>* const dataAssignments,
                              const std::vector<precision>* const weights, const int_size displacement = 0) = 0;

    virtual void normalizeSum(Matrix<precision, int_size>* const avgContainer,
                              const std::vector<precision>* const weightSums) = 0;
};

template <typename precision, typename int_size>
class SerialWeightedMultiVectorAverager : public AbstractWeightedAverager<precision, int_size>
{
public:
    ~SerialWeightedMultiVectorAverager() = default;

    void calculateSum(const Matrix<precision, int_size>* const data, Matrix<precision, int_size>* const avgContainer,
                      const std::vector<int_size>* const dataAssignments, const std::vector<precision>* const weights,
                      const int_size displacement = 0) override;

    void normalizeSum(Matrix<precision, int_size>* const avgContainer,
                      const std::vector<precision>* const weightSums) override;
};

template <typename precision, typename int_size>
class OMPWeightedMultiVectorAverager : public AbstractWeightedAverager<precision, int_size>
{
public:
    ~OMPWeightedMultiVectorAverager() = default;

    void calculateSum(const Matrix<precision, int_size>* const data, Matrix<precision, int_size>* const avgContainer,
                      const std::vector<int_size>* const dataAssignments, const std::vector<precision>* const weights,
                      const int_size displacement = 0) override;

    void normalizeSum(Matrix<precision, int_size>* const avgContainer,
                      const std::vector<precision>* const weightSums) override;
};

template <typename precision, typename int_size>
class AbstractAverager
{
public:
    virtual ~AbstractAverager() = default;

    void calculateAverage(const Matrix<precision, int_size>* const data, std::vector<precision>* const avgContainer);

    virtual void calculateSum(const Matrix<precision, int_size>* const data,
                              std::vector<precision>* const avgContainer) = 0;

    virtual void normalizeSum(std::vector<precision>* const avgContainer, const int_size& numData) = 0;
};

template <typename precision, typename int_size>
class SerialVectorAverager : public AbstractAverager<precision, int_size>
{
public:
    ~SerialVectorAverager() = default;

    void calculateSum(const Matrix<precision, int_size>* const data,
                      std::vector<precision>* const avgContainer) override;

    void normalizeSum(std::vector<precision>* const avgContainer, const int_size& numData) override;
};

template <typename precision, typename int_size>
class OMPVectorAverager : public AbstractAverager<precision, int_size>
{
public:
    ~OMPVectorAverager() = default;

    void calculateSum(const Matrix<precision, int_size>* const data,
                      std::vector<precision>* const avgContainer) override;

    void normalizeSum(std::vector<precision>* const avgContainer, const int_size& numData) override;
};

template <typename precision, typename int_size>
void AbstractWeightedAverager<precision, int_size>::calculateAverage(const Matrix<precision, int_size>* const data,
                                                                     Matrix<precision, int_size>* const avgContainer,
                                                                     const std::vector<int_size>* const dataAssignments,
                                                                     const std::vector<precision>* const weights,
                                                                     const std::vector<precision>* const weightSums,
                                                                     const int_size displacement)
{
    calculateSum(data, avgContainer, dataAssignments, weights, displacement);
    normalizeSum(avgContainer, weightSums);
}

template <typename precision, typename int_size>
void SerialWeightedMultiVectorAverager<precision, int_size>::calculateSum(
  const Matrix<precision, int_size>* const data, Matrix<precision, int_size>* const avgContainer,
  const std::vector<int_size>* const dataAssignments, const std::vector<precision>* const weights,
  const int_size displacement)
{
    for (int32_t i = 0; i < data->size(); ++i)
    {
        for (int32_t j = 0; j < data->cols(); ++j)
        {
            avgContainer->at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
        }
    }
}

template <typename precision, typename int_size>
void SerialWeightedMultiVectorAverager<precision, int_size>::normalizeSum(
  Matrix<precision, int_size>* const avgContainer, const std::vector<precision>* const weightSums)
{
    for (int32_t i = 0; i < avgContainer->size(); ++i)
    {
        for (int32_t j = 0; j < avgContainer->cols(); ++j)
        {
            avgContainer->at(i, j) /= weightSums->at(i);
        }
    }
}

template <typename precision, typename int_size>
void OMPWeightedMultiVectorAverager<precision, int_size>::calculateSum(
  const Matrix<precision, int_size>* const data, Matrix<precision, int_size>* const avgContainer,
  const std::vector<int_size>* const dataAssignments, const std::vector<precision>* const weights,
  const int_size displacement)
{
    for (int32_t i = 0; i < data->size(); ++i)
    {
        for (int32_t j = 0; j < data->cols(); ++j)
        {
            avgContainer->at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
        }
    }

    //         Matrix<precision, int_size>& refContainer = *avgContainer;

    // #pragma omp parallel for shared(data, dataAssignments, weights, displacement), schedule(static), collapse(2),
    // reduction(+ : refContainer)
    //         for (int32_t i = 0; i < data->size(); ++i)
    //         {
    //             for (int32_t j = 0; j < data->cols(); ++j)
    //             {
    //                 refContainer.at(dataAssignments->at(displacement + i), j) += weights->at(i) * data->at(i, j);
    //             }
    //         }
}

template <typename precision, typename int_size>
void OMPWeightedMultiVectorAverager<precision, int_size>::normalizeSum(Matrix<precision, int_size>* const avgContainer,
                                                                       const std::vector<precision>* const weightSums)
{
#pragma omp parallel for schedule(static), collapse(2)
    for (int32_t i = 0; i < avgContainer->size(); ++i)
    {
        for (int32_t j = 0; j < avgContainer->cols(); ++j)
        {
            avgContainer->at(i, j) /= weightSums->at(i);
        }
    }
}

template <typename precision, typename int_size>
void AbstractAverager<precision, int_size>::calculateAverage(const Matrix<precision, int_size>* const data,
                                                             std::vector<precision>* const avgContainer)
{
    calculateSum(data, avgContainer);
    normalizeSum(avgContainer, data->size());
}

template <typename precision, typename int_size>
void SerialVectorAverager<precision, int_size>::calculateSum(const Matrix<precision, int_size>* const data,
                                                             std::vector<precision>* const avgContainer)
{
    for (int32_t i = 0; i < data->size(); ++i)
    {
        for (int32_t j = 0; j < data->cols(); ++j)
        {
            avgContainer->at(j) += data->at(i, j);
        }
    }
}

template <typename precision, typename int_size>
void SerialVectorAverager<precision, int_size>::normalizeSum(std::vector<precision>* const avgContainer,
                                                             const int_size& numData)
{
    std::transform(avgContainer->begin(), avgContainer->end(), avgContainer->begin(),
                   [&numData](const precision& val) { return val / numData; });
}

template <typename precision, typename int_size>
void OMPVectorAverager<precision, int_size>::calculateSum(const Matrix<precision, int_size>* const data,
                                                          std::vector<precision>* const avgContainer)
{
    std::vector<precision>& refContainer = *avgContainer;

#pragma omp parallel for schedule(static), collapse(2), reduction(+ : refContainer)
    for (int32_t i = 0; i < data->size(); ++i)
    {
        for (int32_t j = 0; j < data->cols(); ++j)
        {
            refContainer[j] += data->at(i, j);
        }
    }
}

template <typename precision, typename int_size>
void OMPVectorAverager<precision, int_size>::normalizeSum(std::vector<precision>* const avgContainer,
                                                          const int_size& numData)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < avgContainer->size(); ++i)
    {
        avgContainer->at(i) /= numData;
    }
}
}  // namespace HPKmeans