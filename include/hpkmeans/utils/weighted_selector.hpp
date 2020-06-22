#pragma once

#include <vector>

namespace hpkmeans
{
class WeightedSelector
{
public:
    template <class Container>
    int32_t selectSingle(const Container* const weights, typename Container::value_type randomSumFrac) const
    {
        for (size_t i = 0; i < weights->size(); ++i)
        {
            if ((randomSumFrac -= weights->at(i)) <= 0)
                return i;
        }

        return static_cast<int32_t>(weights->size()) - 1;
    }

    template <class Container>
    std::vector<int32_t> selectMultiple(const Container* const weights, const int32_t sampleSize)
    {
        typedef typename Container::value_type T;

        std::vector<double> vals;
        for (const auto& val : *weights)
        {
            vals.push_back(std::pow(getRandFraction(), 1.0 / val));
        }

        std::vector<std::pair<int32_t, T>> valsWithIndices;
        for (size_t i = 0; i < vals.size(); ++i)
        {
            valsWithIndices.emplace_back(i, vals[i]);
        }
        std::sort(valsWithIndices.begin(), valsWithIndices.end(),
                  [](std::pair<int32_t, T> x, std::pair<int32_t, T> y) { return x.second > y.second; });

        std::vector<int32_t> samples;
        for (int32_t i = 0; i < sampleSize; ++i)
        {
            samples.emplace_back(valsWithIndices[i].first);
        }

        return samples;
    }
};
}  // namespace hpkmeans