#pragma once

#include <hpkmeans/DistanceFunctors.hpp>
#include <hpkmeans/data_types/Matrix.hpp>
#include <memory>
#include <vector>

namespace HPKmeans
{
template <typename precision, typename int_size>
struct ClusterResults
{
    precision error;
    std::shared_ptr<Matrix<precision, int_size>> clusters;
    std::shared_ptr<std::vector<int_size>> clustering;
    std::shared_ptr<std::vector<precision>> clusterWeights;
    std::shared_ptr<std::vector<precision>> sqDistances;

    ClusterResults() noexcept : error(-1.0) {}

    ClusterResults(ClusterResults&& other) noexcept : error(-1.0) { *this = std::move(other); }

    ~ClusterResults() = default;

    ClusterResults& operator=(ClusterResults&& rhs)
    {
        if (this != &rhs)
        {
            error          = rhs.error;
            clusters       = std::move(rhs.clusters);
            clustering     = std::move(rhs.clustering);
            clusterWeights = std::move(rhs.clusterWeights);
            sqDistances    = std::move(rhs.sqDistances);
        }

        return *this;
    }
};

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