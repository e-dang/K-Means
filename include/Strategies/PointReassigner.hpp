#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class AbstractPointReassigner
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> pUpdater;

public:
    AbstractPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) : pUpdater(updater){};

    virtual ~AbstractPointReassigner() = default;

    int_size reassignPoint(const int_size& dataIdx, KmeansData<precision, int_size>* const kmeansData);

    virtual int_size reassignPoints(KmeansData<precision, int_size>* const kmeansData) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    SerialPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~SerialPointReassigner() = default;

    int_size reassignPoints(KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision = double, typename int_size = int32_t>
class SerialOptimizedPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    SerialOptimizedPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~SerialOptimizedPointReassigner() = default;

    int_size reassignPoints(KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision = double, typename int_size = int32_t>
class OMPPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    OMPPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~OMPPointReassigner() = default;

    int_size reassignPoints(KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision = double, typename int_size = int32_t>
class OMPOptimizedPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    OMPOptimizedPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~OMPOptimizedPointReassigner() = default;

    int_size reassignPoints(KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
int_size AbstractPointReassigner<precision, int_size>::reassignPoint(const int_size& dataIdx,
                                                                     KmeansData<precision, int_size>* const kmeansData)
{
    auto before = kmeansData->clusteringAt(dataIdx);

    pUpdater->findAndUpdateClosestCluster(dataIdx, kmeansData);

    if (before != kmeansData->clusteringAt(dataIdx))
    {
        return 1;
    }
    return 0;
}

template <typename precision, typename int_size>
int_size SerialPointReassigner<precision, int_size>::reassignPoints(KmeansData<precision, int_size>* const kmeansData)
{
    int_size changed = 0;
    for (int_size i = 0; i < kmeansData->data->size(); i++)
    {
        changed += this->reassignPoint(i, kmeansData);
    }

    return changed;
}

template <typename precision, typename int_size>
int_size SerialOptimizedPointReassigner<precision, int_size>::reassignPoints(
  KmeansData<precision, int_size>* const kmeansData)
{
    int_size changed = 0;
    auto numFeatures = kmeansData->data->cols();

    for (int_size i = 0; i < kmeansData->data->size(); i++)
    {
        auto clusterIdx = kmeansData->clusteringAt(i);
        auto dist       = std::pow(
          (*kmeansData->distanceFunc)(kmeansData->data->at(i), kmeansData->clusters->at(clusterIdx), numFeatures), 2);
        if (dist > kmeansData->sqDistancesAt(i) || kmeansData->sqDistancesAt(i) < 0)
        {
            changed += this->reassignPoint(i, kmeansData);
        }
        else
        {
            kmeansData->sqDistancesAt(i) = dist;
        }
    }

    return changed;
}

template <typename precision, typename int_size>
int_size OMPPointReassigner<precision, int_size>::reassignPoints(KmeansData<precision, int_size>* const kmeansData)
{
    int_size changed = 0;

#pragma omp parallel for schedule(static), reduction(+ : changed)
    for (int_size i = 0; i < kmeansData->data->size(); i++)
    {
        changed += this->reassignPoint(i, kmeansData);
    }

    return changed;
}

template <typename precision, typename int_size>
int_size OMPOptimizedPointReassigner<precision, int_size>::reassignPoints(
  KmeansData<precision, int_size>* const kmeansData)
{
    int_size changed = 0;
    auto numFeatures = kmeansData->data->cols();

#pragma omp parallel for shared(numFeatures), schedule(static), reduction(+ : changed)
    for (int_size i = 0; i < kmeansData->data->size(); i++)
    {
        auto clusterIdx = kmeansData->clusteringAt(i);
        auto dist       = std::pow(
          (*kmeansData->distanceFunc)(kmeansData->data->at(i), kmeansData->clusters->at(clusterIdx), numFeatures), 2);
        if (dist > kmeansData->sqDistancesAt(i) || kmeansData->sqDistancesAt(i) < 0)
        {
            changed += this->reassignPoint(i, kmeansData);
        }
        else
        {
            kmeansData->sqDistancesAt(i) = dist;
        }
    }

    return changed;
}
}  // namespace HPKmeans