#pragma once

#include <hpkmeans/data_types/matrix.hpp>
#include <memory>
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

enum Initializer
{
    InitNull = 0,
    KPP      = 1 << 0,
    OptKPP   = 1 << 1
};

enum Maximizer
{
    MaxNull  = 0,
    Lloyd    = 1 << 2,
    OptLloyd = 1 << 3
};

enum CoresetCreator
{
    None      = 0,
    LWCoreset = 1 << 4,
};

enum Parallelism
{
    ParaNull = 0,
    Serial   = 1 << 5,
    OMP      = 1 << 6,
    MPI      = 1 << 7,
    Hybrid   = 1 << 8
};

enum Variant
{
    Reg,
    Opt,
    SpecificCoreset
};
}  // namespace HPKmeans