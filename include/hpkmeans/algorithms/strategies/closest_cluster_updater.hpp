#pragma once

#include <hpkmeans/algorithms/strategies/closest_cluster_finder.hpp>
#include <hpkmeans/algorithms/strategies/clustering_data_updater.hpp>
#include <hpkmeans/data_types/kmeans_state.hpp>
#include <memory>

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractClosestClusterUpdater
{
protected:
    std::unique_ptr<IClosestClusterFinder<precision, int_size>> pFinder;
    std::unique_ptr<AbstractClusteringDataUpdater<precision, int_size>> pUpdater;

public:
    AbstractClosestClusterUpdater(IClosestClusterFinder<precision, int_size>* finder,
                                  AbstractClusteringDataUpdater<precision, int_size>* updater) :
        pFinder(finder), pUpdater(updater)
    {
    }

    virtual ~AbstractClosestClusterUpdater() = default;

    void findAndUpdateClosestCluster(const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState);

    virtual void findAndUpdateClosestClusters(KmeansState<precision, int_size>* const kmeansState) = 0;
};

template <typename precision, typename int_size>
class SerialClosestClusterUpdater : public AbstractClosestClusterUpdater<precision, int_size>
{
public:
    SerialClosestClusterUpdater(IClosestClusterFinder<precision, int_size>* finder,
                                AbstractClusteringDataUpdater<precision, int_size>* updater) :
        AbstractClosestClusterUpdater<precision, int_size>(finder, updater)
    {
    }

    ~SerialClosestClusterUpdater() = default;

    void findAndUpdateClosestClusters(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class OMPClosestClusterUpdater : public AbstractClosestClusterUpdater<precision, int_size>
{
public:
    OMPClosestClusterUpdater(IClosestClusterFinder<precision, int_size>* finder,
                             AbstractClusteringDataUpdater<precision, int_size>* updater) :
        AbstractClosestClusterUpdater<precision, int_size>(finder, updater)
    {
    }

    ~OMPClosestClusterUpdater() = default;

    void findAndUpdateClosestClusters(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
void AbstractClosestClusterUpdater<precision, int_size>::findAndUpdateClosestCluster(
  const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState)
{
    auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansState);
    pUpdater->update(dataIdx, closestCluster, kmeansState);
}

template <typename precision, typename int_size>
void SerialClosestClusterUpdater<precision, int_size>::findAndUpdateClosestClusters(
  KmeansState<precision, int_size>* const kmeansState)
{
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        this->findAndUpdateClosestCluster(i, kmeansState);
    }
}

template <typename precision, typename int_size>
void OMPClosestClusterUpdater<precision, int_size>::findAndUpdateClosestClusters(
  KmeansState<precision, int_size>* const kmeansState)
{
#pragma omp parallel for schedule(static)
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        this->findAndUpdateClosestCluster(i, kmeansState);
    }
}
}  // namespace HPKmeans