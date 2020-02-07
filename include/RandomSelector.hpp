#pragma once

#include <vector>

#include "Definitions.hpp"

class IWeightedRandomSelector
{
public:
    virtual int select(const std::vector<value_t>* const weights, value_t randomSumFrac) = 0;
};

class SingleWeightedRandomSelector : public IWeightedRandomSelector
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
    int select(const std::vector<value_t>* const weights, value_t randomSumFrac) override;
};

class IMultiWeightedRandomSelector
{
public:
    virtual std::vector<int> select(const std::vector<value_t>* const weights, const int_fast32_t& sampleSize) = 0;
};

/**
 * @brief Class to perform random weighted selection from a list without replacement. This algorithm was taken from
 *        https://stackoverflow.com/questions/53632441/c-sampling-from-discrete-distribution-without-replacement
 *
 */
class MultiWeightedRandomSelector : public IMultiWeightedRandomSelector
{
public:
    std::vector<int> select(const std::vector<value_t>* const weights, const int_fast32_t& sampleSize) override;
};