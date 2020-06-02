#pragma once

#include <vector>

namespace hpkmeans
{
template <typename T>
class WeightedSelector
{
public:
    int32_t select(const std::vector<T>* const weights, T randomSumFrac) const
    {
        for (size_t i = 0; i < weights->size(); ++i)
        {
            if ((randomSumFrac -= weights->at(i)) <= 0)
            {
                return i;
            }
        }

        return static_cast<int32_t>(weights->size()) - 1;
    }
};
}  // namespace hpkmeans