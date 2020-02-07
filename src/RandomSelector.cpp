#include "RandomSelector.hpp"

#include "Utils.hpp"

int_fast32_t SingleWeightedRandomSelector::select(const std::vector<value_t>* const weights, value_t randomSumFrac)
{
    for (size_t i = 0; i < weights->size(); i++)
    {
        if ((randomSumFrac -= weights->at(i)) <= 0)
        {
            return i;
        }
    }

    return static_cast<int_fast32_t>(weights->size()) - 1;
}

std::vector<int_fast32_t> MultiWeightedRandomSelector::select(const std::vector<value_t>* const weights,
                                                              const int_fast32_t& sampleSize)
{
    std::vector<double> vals;
    for (const auto& val : *weights)
    {
        vals.push_back(std::pow(getRandDouble01(), 1.0 / val));
    }

    std::vector<std::pair<int_fast32_t, value_t>> valsWithIndices;
    for (size_t i = 0; i < vals.size(); i++)
    {
        valsWithIndices.emplace_back(i, vals[i]);
    }
    std::sort(
      valsWithIndices.begin(), valsWithIndices.end(),
      [](std::pair<int_fast32_t, value_t> x, std::pair<int_fast32_t, value_t> y) { return x.second > y.second; });

    std::vector<int_fast32_t> samples;
    for (int_fast32_t i = 0; i < sampleSize; i++)
    {
        samples.emplace_back(valsWithIndices[i].first);
    }

    return samples;
}