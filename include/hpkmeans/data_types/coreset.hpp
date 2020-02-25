#pragma once

#include <hpkmeans/data_types/matrix.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
struct Coreset
{
    Matrix<precision, int_size> data;
    std::vector<precision> weights;

    Coreset(const int_size& numData, const int_size& numFeatures, bool autoReserve = false) :
        data(numData, numFeatures, autoReserve)
    {
        if (autoReserve)
        {
            weights = std::vector<precision>(numData);
        }
        else
        {
            weights.reserve(numData);
        }
    }

    Coreset(Coreset&& other) : data(), weights() { *this = std::move(other); }

    ~Coreset() = default;

    Coreset& operator=(Coreset&& rhs)
    {
        if (this != &rhs)
        {
            data    = std::move(rhs.data);
            weights = std::move(rhs.weights);
        }

        return *this;
    }
};
}  // namespace HPKmeans