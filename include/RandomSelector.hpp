#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

#include "Definitions.hpp"

class AbstractWeightedRandomSelector
{
public:
    virtual int select(std::vector<value_t>* weights, value_t randomSumFrac) = 0;
};

class SingleWeightedRandomSelector : public AbstractWeightedRandomSelector
{
public:
    /**
     * @brief Algorithm for a weighted random selection of an index in the range of [0, maxIdx), where maxIdx is the
     * length of the weights vector. The algorithm works by summing the weights and multiplying the result by a random
     * float that is [0, 1). Then we sequentially subtract each weight from the random sum fraction until the value is
     * less than or equal to 0 at which point the index of the weight that turned the sum for positive to <= 0 is
     *        returned.
     *
     * @param weights - The weights of the indices to select from.
     * @param randomFrac - A random float used in the algorithm to perform the random selection, [0, 1).
     * @return int - The selected index.
     */
    int select(std::vector<value_t>* weights, value_t randomSumFrac) override;
};

class IMultiWeightedRandomSelector
{
public:
    virtual std::vector<int> select(std::vector<value_t>* weights, const int& sampleSize) = 0;
};

class MultiWeightedRandomSelector : public IMultiWeightedRandomSelector
{
public:
    std::vector<int> select(std::vector<value_t>* weights, const int& sampleSize) override;
};