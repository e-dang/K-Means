#pragma once

#include <cstdint>
#include <hpkmeans/utils/utils.hpp>
#include <vector>

namespace HPKmeans
{
template <typename precision, typename int_size>
class IWeightedRandomSelector
{
public:
    virtual ~IWeightedRandomSelector() = default;

    virtual int_size select(const std::vector<precision>* const weights, precision randomSumFrac) = 0;
};

template <typename precision, typename int_size>
class SingleWeightedRandomSelector : public IWeightedRandomSelector<precision, int_size>
{
public:
    ~SingleWeightedRandomSelector() = default;

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
    int_size select(const std::vector<precision>* const weights, precision randomSumFrac) override;
};

template <typename precision, typename int_size>
class IMultiWeightedRandomSelector
{
public:
    virtual ~IMultiWeightedRandomSelector() = default;

    virtual std::vector<int_size> select(const std::vector<precision>* const weights, const int_size& sampleSize) = 0;
};

/**
 * @brief Class to perform random weighted selection from a list without replacement. This algorithm was taken from
 *        https://stackoverflow.com/questions/53632441/c-sampling-from-discrete-distribution-without-replacement
 *
 */
template <typename precision, typename int_size>
class MultiWeightedRandomSelector : public IMultiWeightedRandomSelector<precision, int_size>
{
public:
    std::vector<int_size> select(const std::vector<precision>* const weights, const int_size& sampleSize) override;
};

template <typename precision, typename int_size>
int_size SingleWeightedRandomSelector<precision, int_size>::select(const std::vector<precision>* const weights,
                                                                   precision randomSumFrac)
{
    for (size_t i = 0; i < weights->size(); ++i)
    {
        if ((randomSumFrac -= weights->at(i)) <= 0)
        {
            return i;
        }
    }

    return static_cast<int_size>(weights->size()) - 1;
}

template <typename precision, typename int_size>
std::vector<int_size> MultiWeightedRandomSelector<precision, int_size>::select(
  const std::vector<precision>* const weights, const int_size& sampleSize)
{
    std::vector<double> vals;
    for (const auto& val : *weights)
    {
        vals.push_back(std::pow(getRandDouble01(), 1.0 / val));
    }

    std::vector<std::pair<int_size, precision>> valsWithIndices;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        valsWithIndices.emplace_back(i, vals[i]);
    }
    std::sort(valsWithIndices.begin(), valsWithIndices.end(),
              [](std::pair<int_size, precision> x, std::pair<int_size, precision> y) { return x.second > y.second; });

    std::vector<int_size> samples;
    for (int_size i = 0; i < sampleSize; ++i)
    {
        samples.emplace_back(valsWithIndices[i].first);
    }

    return samples;
}
}  // namespace HPKmeans