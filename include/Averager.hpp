#pragma once

#include "Definitions.hpp"
#include "DataClasses.hpp"

class AbstractWeightedAverager
{
public:
    virtual void calculateSum(Matrix *data, Matrix *avgContainer, std::vector<int> *dataAssignments,
                              std::vector<value_t> *weights) = 0;
    virtual void normalizeSum(Matrix *avgContainer, std::vector<value_t> *weightSums) = 0;
    virtual void calculateAverage(Matrix *data, Matrix *avgContainer, std::vector<int> *dataAssignments,
                                  std::vector<value_t> *weights, std::vector<value_t> *weightSums) = 0;
};

class WeightedMultiVectorAverager : public AbstractWeightedAverager
{
public:
    void calculateSum(Matrix *data, Matrix *avgContainer, std::vector<int> *dataAssignments,
                      std::vector<value_t> *weights) override;

    void normalizeSum(Matrix *avgContainer, std::vector<value_t> *weightSums) override;

    void calculateAverage(Matrix *data, Matrix *avgContainer, std::vector<int> *dataAssignments,
                          std::vector<value_t> *weights, std::vector<value_t> *weightSums) override;
};

class OMPWeightedMultiVectorAverager : public WeightedMultiVectorAverager
{
public:
    void normalizeSum(Matrix *avgContainer, std::vector<value_t> *weightSums) override;
};

class AbstractAverager
{
public:
    virtual void calculateSum(Matrix *data, std::vector<value_t> *avgContainer) = 0;
    virtual void normalizeSum(std::vector<value_t> *avgContainer, int numData) = 0;
    virtual void calculateAverage(Matrix *data, std::vector<value_t> *avgContainer) = 0;
};

class VectorAverager : public AbstractAverager
{
public:
    void calculateSum(Matrix *data, std::vector<value_t> *avgContainer) override;
    void normalizeSum(std::vector<value_t> *avgContainer, int numData) override;
    void calculateAverage(Matrix *data, std::vector<value_t> *avgContainer) override;
};