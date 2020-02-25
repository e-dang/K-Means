#pragma once

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
}