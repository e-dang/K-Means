#include "RandomSelector.hpp"

#include "Utils.hpp"

int SingleWeightedRandomSelector::select(const std::vector<value_t>* const weights, value_t randomSumFrac)
{
    int maxIdx = weights->size();

    // each iteration substract the weight from the cutoff and once it reaches <= 0, the corresponding index is selected
    for (int i = 0; i < maxIdx; i++)
    {
        if ((randomSumFrac -= weights->at(i)) <= 0)
        {
            return i;
        }
    }

    return maxIdx - 1;
}

std::vector<int> MultiWeightedRandomSelector::select(const std::vector<value_t>* const weights,
                                                     const size_t& sampleSize)
{
    std::vector<double> vals;
    for (auto& val : *weights)
    {
        vals.push_back(std::pow(getRandDouble01(), 1.0 / val));
    }

    std::vector<std::pair<int, value_t>> valsWithIndices;
    for (int i = 0; i < vals.size(); i++)
    {
        valsWithIndices.emplace_back(i, vals[i]);
    }
    std::sort(valsWithIndices.begin(), valsWithIndices.end(),
              [](std::pair<int, value_t> x, std::pair<int, value_t> y) { return x.second > y.second; });

    std::vector<int> samples;
    for (size_t i = 0; i < sampleSize; i++)
    {
        samples.push_back(valsWithIndices[i].first);
    }

    return samples;
}