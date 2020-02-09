#include "Strategies/RandomSelector.hpp"

#include "Utils/Utils.hpp"

int32_t SingleWeightedRandomSelector::select(const std::vector<value_t>* const weights, value_t randomSumFrac)
{
    for (size_t i = 0; i < weights->size(); i++)
    {
        if ((randomSumFrac -= weights->at(i)) <= 0)
        {
            return i;
        }
    }

    return static_cast<int32_t>(weights->size()) - 1;
}

std::vector<int32_t> MultiWeightedRandomSelector::select(const std::vector<value_t>* const weights,
                                                         const int32_t& sampleSize)
{
    std::vector<double> vals;
    for (const auto& val : *weights)
    {
        vals.push_back(std::pow(getRandDouble01(), 1.0 / val));
    }

    std::vector<std::pair<int32_t, value_t>> valsWithIndices;
    for (size_t i = 0; i < vals.size(); i++)
    {
        valsWithIndices.emplace_back(i, vals[i]);
    }
    std::sort(valsWithIndices.begin(), valsWithIndices.end(),
              [](std::pair<int32_t, value_t> x, std::pair<int32_t, value_t> y) { return x.second > y.second; });

    std::vector<int32_t> samples;
    for (int32_t i = 0; i < sampleSize; i++)
    {
        samples.emplace_back(valsWithIndices[i].first);
    }

    return samples;
}