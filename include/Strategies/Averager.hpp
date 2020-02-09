#pragma once

#include "Containers/DataClasses.hpp"

class AbstractWeightedAverager
{
public:
    void calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                          const std::vector<int32_t>* const dataAssignments, const std::vector<value_t>* const weights,
                          const std::vector<value_t>* const weightSums, const int32_t displacement = 0);

    virtual void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                              const std::vector<int32_t>* const dataAssignments,
                              const std::vector<value_t>* const weights, const int32_t displacement = 0) = 0;

    virtual void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) = 0;
};

class SerialWeightedMultiVectorAverager : public AbstractWeightedAverager
{
public:
    void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                      const std::vector<int32_t>* const dataAssignments, const std::vector<value_t>* const weights,
                      const int32_t displacement = 0) override;

    void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) override;
};

class OMPWeightedMultiVectorAverager : public AbstractWeightedAverager
{
public:
    void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                      const std::vector<int32_t>* const dataAssignments, const std::vector<value_t>* const weights,
                      const int32_t displacement = 0) override;

    void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) override;
};

class AbstractAverager
{
public:
    void calculateAverage(const Matrix* const data, std::vector<value_t>* const avgContainer);

    virtual void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) = 0;

    virtual void normalizeSum(std::vector<value_t>* const avgContainer, const int32_t& numData) = 0;
};

class SerialVectorAverager : public AbstractAverager
{
public:
    void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) override;

    void normalizeSum(std::vector<value_t>* const avgContainer, const int32_t& numData) override;
};

class OMPVectorAverager : public AbstractAverager
{
public:
    void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) override;

    void normalizeSum(std::vector<value_t>* const avgContainer, const int32_t& numData) override;
};