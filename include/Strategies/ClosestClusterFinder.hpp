#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class IClosestClusterFinder
{
public:
    virtual ~IClosestClusterFinder() = default;

    virtual ClosestCluster<precision, int_size> findClosestCluster(
      const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class ClosestClusterFinder : public IClosestClusterFinder<precision, int_size>
{
public:
    ~ClosestClusterFinder() = default;

    ClosestCluster<precision, int_size> findClosestCluster(const int_size& dataIdx,
                                                           KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision = double, typename int_size = int32_t>
class ClosestNewClusterFinder : public IClosestClusterFinder<precision, int_size>
{
public:
    ~ClosestNewClusterFinder() = default;

    ClosestCluster<precision, int_size> findClosestCluster(const int_size& dataIdx,
                                                           KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
ClosestCluster<precision, int_size> ClosestClusterFinder<precision, int_size>::findClosestCluster(
  const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData)
{
    auto numFeatures         = kmeansData->data->cols();
    auto numExistingClusters = kmeansData->clusters->size();
    const auto datapoint     = kmeansData->data->at(dataIdx);
    precision minDistance    = -1.0;
    int_size clusterIdx      = -1;

    for (int32_t i = 0; i < numExistingClusters; i++)
    {
        precision tempDistance = (*kmeansData->distanceFunc)(datapoint, kmeansData->clusters->at(i), numFeatures);

        if (minDistance > tempDistance || minDistance < 0)
        {
            minDistance = tempDistance;
            clusterIdx  = i;
        }
    }

    return ClosestCluster<precision, int_size>{ clusterIdx, std::pow(minDistance, 2) };
}

template <typename precision, typename int_size>
ClosestCluster<precision, int_size> ClosestNewClusterFinder<precision, int_size>::findClosestCluster(
  const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData)
{
    thread_local static int_size prevNumClusters;
    thread_local static int_size intermediate;
    auto numFeatures         = kmeansData->data->cols();
    auto numExistingClusters = kmeansData->clusters->size();
    const auto datapoint     = kmeansData->data->at(dataIdx);
    precision minDistance    = -1.0;
    int_size clusterIdx      = -1;

    if (numExistingClusters == 1)
    {
        prevNumClusters = 0;
        intermediate    = 1;
    }

    for (auto i = prevNumClusters; i < numExistingClusters; i++)
    {
        precision tempDistance = (*kmeansData->distanceFunc)(datapoint, kmeansData->clusters->at(i), numFeatures);

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

    return ClosestCluster<precision, int_size>{ clusterIdx, std::pow(minDistance, 2) };
}
}  // namespace HPKmeans