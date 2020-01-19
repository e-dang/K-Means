#include "Utils.hpp"
#include <numeric>

int weightedRandomSelection(std::vector<value_t> *weights, float randomFrac)
{
    int maxIdx = weights->size();
    value_t cutoff = randomFrac * std::accumulate(weights->begin(), weights->end(), 0);

    // each iteration substract the weight from the cutoff and once it reaches <= 0, the corresponding index is selected
    for (int i = 0; i < maxIdx; i++)
    {
        if ((cutoff -= weights->at(i)) <= 0)
        {
            return i;
        }
    }

    return maxIdx;
}