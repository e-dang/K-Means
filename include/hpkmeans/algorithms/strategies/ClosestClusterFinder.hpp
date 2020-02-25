#pragma once

#include <hpkmeans/data_types/KmeansState.hpp>

namespace HPKmeans
{
/**
 * @brief A return structure that couples the distance between a point and its closest cluster and the index of that
 *        cluster together.
 */
template <typename precision, typename int_size>
struct ClosestCluster
{
    int_size clusterIdx;
    precision distance;
};

template <typename precision, typename int_size>
class IClosestClusterFinder
{
public:
    virtual ~IClosestClusterFinder() = default;

    virtual ClosestCluster<precision, int_size> findClosestCluster(
      const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState) = 0;
};

template <typename precision, typename int_size>
class ClosestClusterFinder : public IClosestClusterFinder<precision, int_size>
{
public:
    ~ClosestClusterFinder() = default;

    ClosestCluster<precision, int_size> findClosestCluster(
      const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class ClosestNewClusterFinder : public IClosestClusterFinder<precision, int_size>
{
public:
    ~ClosestNewClusterFinder() = default;

    ClosestCluster<precision, int_size> findClosestCluster(
      const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
ClosestCluster<precision, int_size> ClosestClusterFinder<precision, int_size>::findClosestCluster(
  const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState)
{
    auto numExistingClusters = kmeansState->clustersSize();
    const auto datapoint     = kmeansState->dataAt(dataIdx);
    precision minDistance    = -1.0;
    int_size clusterIdx      = -1;

    for (int32_t i = 0; i < numExistingClusters; ++i)
    {
        precision tempDistance = (*kmeansState)(datapoint, kmeansState->clustersAt(i));

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    return ClosestCluster<precision, int_size>{ clusterIdx, static_cast<precision>(std::pow(minDistance, 2)) };
}

template <typename precision, typename int_size>
ClosestCluster<precision, int_size> ClosestNewClusterFinder<precision, int_size>::findClosestCluster(
  const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState)
{
    thread_local static int_size prevNumClusters;
    thread_local static int_size intermediate;
    auto numExistingClusters = kmeansState->clustersSize();
    const auto datapoint     = kmeansState->dataAt(dataIdx);
    precision minDistance    = -1.0;
    int_size clusterIdx      = -1;

    if (numExistingClusters == 1)
    {
        prevNumClusters = 0;
        intermediate    = 1;
    }

    for (auto i = prevNumClusters; i < numExistingClusters; ++i)
    {
        precision tempDistance = (*kmeansState)(datapoint, kmeansState->clustersAt(i));

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    if (intermediate != numExistingClusters)
    {
        prevNumClusters = intermediate;
        intermediate    = numExistingClusters;
    }

    return ClosestCluster<precision, int_size>{ clusterIdx, static_cast<precision>(std::pow(minDistance, 2)) };
}
}  // namespace HPKmeans