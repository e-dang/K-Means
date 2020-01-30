#include "RandomSelector.hpp"

int SingleWeightedRandomSelector::select(std::vector<value_t> *weights, value_t randomSumFrac)
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