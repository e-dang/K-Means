#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"

class AbstractWeightedAverager
{
public:
    virtual void calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                                  const std::vector<int>* const dataAssignments,
                                  const std::vector<value_t>* const weights,
                                  const std::vector<value_t>* const weightSums) = 0;

    virtual void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                              const std::vector<int>* const dataAssignments,
                              const std::vector<value_t>* const weights) = 0;

    virtual void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) = 0;
};

class WeightedMultiVectorAverager : public AbstractWeightedAverager
{
public:
    void calculateAverage(const Matrix* const data, Matrix* const avgContainer,
                          const std::vector<int>* const dataAssignments, const std::vector<value_t>* const weights,
                          const std::vector<value_t>* const weightSums) override;

    void calculateSum(const Matrix* const data, Matrix* const avgContainer,
                      const std::vector<int>* const dataAssignments,
                      const std::vector<value_t>* const weights) override;

    void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) override;
};

class OMPWeightedMultiVectorAverager : public WeightedMultiVectorAverager
{
public:
    void normalizeSum(Matrix* const avgContainer, const std::vector<value_t>* const weightSums) override;
};

class AbstractAverager
{
public:
    void calculateAverage(const Matrix* const data, std::vector<value_t>* const avgContainer);

    virtual void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) = 0;

    virtual void normalizeSum(std::vector<value_t>* const avgContainer, const int numData) = 0;
};

class VectorAverager : public AbstractAverager
{
public:
    void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) override;

    void normalizeSum(std::vector<value_t>* const avgContainer, const int numData) override;
};

class OMPVectorAverager : public AbstractAverager
{
public:
    void calculateSum(const Matrix* const data, std::vector<value_t>* const avgContainer) override;

    void normalizeSum(std::vector<value_t>* const avgContainer, const int numData) override;
};