#pragma once

#include <vector>

namespace HPKmeans
{
template <typename int_size>
struct MPIData
{
    int rank;
    int numProcs;
    std::vector<int_size> lengths;
    std::vector<int_size> displacements;

    MPIData(const int_size& rank, const int_size& numProcs, const std::vector<int_size>& lengths,
            const std::vector<int_size>& displacements) :
        rank(rank), numProcs(numProcs), lengths(lengths), displacements(displacements)
    {
    }

    ~MPIData() = default;

    MPIData(MPIData&& other) : rank(0), numProcs(0), lengths(), displacements() { *this = std::move(other); }

    MPIData& operator=(MPIData&& rhs)
    {
        if (this != &rhs)
        {
            rank          = rhs.rank;
            numProcs      = rhs.numProcs;
            lengths       = std::move(rhs.lengths);
            displacements = std::move(rhs.displacements);
        }

        return *this;
    }
};
}  // namespace HPKmeans