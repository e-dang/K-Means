#pragma once

#include <memory>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterFinder.hpp"
#include "Strategies/ClusteringUpdater.hpp"

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
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

    void findAndUpdateClosestCluster(const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData)
    {
        auto closestCluster = pFinder->findClosestCluster(dataIdx, kmeansData);
        pUpdater->update(dataIdx, closestCluster, kmeansData);
    }

    virtual void findAndUpdateClosestClusters(KmeansData<precision, int_size>* const kmeansData) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialClosestClusterUpdater : public AbstractClosestClusterUpdater<precision, int_size>
{
public:
    SerialClosestClusterUpdater(IClosestClusterFinder<precision, int_size>* finder,
                                AbstractClusteringDataUpdater<precision, int_size>* updater) :
        AbstractClosestClusterUpdater<precision, int_size>(finder, updater)
    {
    }

    ~SerialClosestClusterUpdater() = default;

    void findAndUpdateClosestClusters(KmeansData<precision, int_size>* const kmeansData) override
    {
        for (int_size i = 0; i < kmeansData->data->size(); i++)
        {
            this->findAndUpdateClosestCluster(i, kmeansData);
        }
    }
};

template <typename precision = double, typename int_size = int32_t>
class OMPClosestClusterUpdater : public AbstractClosestClusterUpdater<precision, int_size>
{
public:
    OMPClosestClusterUpdater(IClosestClusterFinder<precision, int_size>* finder,
                             AbstractClusteringDataUpdater<precision, int_size>* updater) :
        AbstractClosestClusterUpdater<precision, int_size>(finder, updater)
    {
    }

    ~OMPClosestClusterUpdater() = default;

    void findAndUpdateClosestClusters(KmeansData<precision, int_size>* const kmeansData) override
    {
#pragma omp parallel for schedule(static)
        for (int_size i = 0; i < kmeansData->data->size(); i++)
        {
            this->findAndUpdateClosestCluster(i, kmeansData);
        }
    }
};
}  // namespace HPKmeans