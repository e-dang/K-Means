#pragma once

#include <memory>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterFinder.hpp"
#include "Strategies/ClusteringUpdater.hpp"

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

    void findAndUpdateClosestCluster(const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData);

    virtual void findAndUpdateClosestClusters(KmeansData<precision, int_size>* const kmeansData) = 0;
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

    void findAndUpdateClosestClusters(KmeansData<precision, int_size>* const kmeansData) override;
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

    void findAndUpdateClosestClusters(KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
void AbstractClosestClusterUpdater<precision, int_size>::findAndUpdateClosestCluster(
  const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData)
{
    auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansData);
    pUpdater->update(dataIdx, closestCluster, kmeansData);
}

template <typename precision, typename int_size>
void SerialClosestClusterUpdater<precision, int_size>::findAndUpdateClosestClusters(
  KmeansData<precision, int_size>* const kmeansData)
{
    for (int_size i = 0; i < kmeansData->dataSize(); ++i)
    {
        this->findAndUpdateClosestCluster(i, kmeansData);
    }
}

template <typename precision, typename int_size>
void OMPClosestClusterUpdater<precision, int_size>::findAndUpdateClosestClusters(
  KmeansData<precision, int_size>* const kmeansData)
{
#pragma omp parallel for schedule(static)
    for (int_size i = 0; i < kmeansData->data->size(); ++i)
    {
        this->findAndUpdateClosestCluster(i, kmeansData);
    }
}
}  // namespace HPKmeans