#pragma once

#include "Containers/DataClasses.hpp"

class AbstractWeightedAverager
{
public:
    virtual void calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                                  const std::vector<int_fast32_t>* const dataAssignments,
                                  const std::vector<value_t>* const weights,
                                  const std::vector<value_t>* const weightSums) = 0;

    virtual void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                              const std::vector<int_fast32_t>* const dataAssignments,
                              const std::vector<value_t>* const weights) = 0;

    virtual void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) = 0;
};

class SerialWeightedMultiVectorAverager : public AbstractWeightedAverager
{
public:
    void calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                          const std::vector<int_fast32_t>* const dataAssignments,
                          const std::vector<value_t>* const weights,
                          const std::vector<value_t>* const weightSums) override;

    void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                      const std::vector<int_fast32_t>* const dataAssignments,
                      const std::vector<value_t>* const weights) override;

    void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) override;
};

class OMPWeightedMultiVectorAverager : public SerialWeightedMultiVectorAverager
{
public:
    void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) override;
};

class AbstractAverager
{
public:
    void calculateAverage(const Matrix* const data, std::vector<value_t>* const avgContainer);

    virtual void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) = 0;

    virtual void normalizeSum(std::vector<value_t>* const avgContainer, const int_fast32_t& numData) = 0;
};

class SerialVectorAverager : public AbstractAverager
{
public:
    void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) override;

    void normalizeSum(std::vector<value_t>* const avgContainer, const int_fast32_t& numData) override;
};

class OMPVectorAverager : public AbstractAverager
{
public:
    void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) override;

    void normalizeSum(std::vector<value_t>* const avgContainer, const int_fast32_t& numData) override;
};