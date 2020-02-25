#pragma once

#include <hpkmeans/algorithms/strategies/ClosestClusterUpdater.hpp>
#include <hpkmeans/data_types/kmeans_state.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractPointReassigner
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> pUpdater;

public:
    AbstractPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) : pUpdater(updater){};

    virtual ~AbstractPointReassigner() = default;

    int_size reassignPoint(const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState);

    virtual int_size reassignPoints(KmeansState<precision, int_size>* const kmeansState) = 0;
};

template <typename precision, typename int_size>
class SerialPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    SerialPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~SerialPointReassigner() = default;

    int_size reassignPoints(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class SerialOptimizedPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    SerialOptimizedPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~SerialOptimizedPointReassigner() = default;

    int_size reassignPoints(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class OMPPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    OMPPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~OMPPointReassigner() = default;

    int_size reassignPoints(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class OMPOptimizedPointReassigner : public AbstractPointReassigner<precision, int_size>
{
public:
    OMPOptimizedPointReassigner(AbstractClosestClusterUpdater<precision, int_size>* updater) :
        AbstractPointReassigner<precision, int_size>(updater)
    {
    }

    ~OMPOptimizedPointReassigner() = default;

    int_size reassignPoints(KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
int_size AbstractPointReassigner<precision, int_size>::reassignPoint(
  const int_size& dataIdx, KmeansState<precision, int_size>* const kmeansState)
{
    auto before = kmeansState->clusteringAt(dataIdx);

    pUpdater->findAndUpdateClosestCluster(dataIdx, kmeansState);

    if (before != kmeansState->clusteringAt(dataIdx))
    {
        return 1;
    }
    return 0;
}

template <typename precision, typename int_size>
int_size SerialPointReassigner<precision, int_size>::reassignPoints(KmeansState<precision, int_size>* const kmeansState)
{
    int_size changed = 0;
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        changed += this->reassignPoint(i, kmeansState);
    }

    return changed;
}

template <typename precision, typename int_size>
int_size SerialOptimizedPointReassigner<precision, int_size>::reassignPoints(
  KmeansState<precision, int_size>* const kmeansState)
{
    int_size changed = 0;

    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        auto clusterIdx = kmeansState->clusteringAt(i);
        auto dist       = std::pow((*kmeansState)(kmeansState->dataAt(i), kmeansState->clustersAt(clusterIdx)), 2);
        if (dist > kmeansState->sqDistancesAt(i) || kmeansState->sqDistancesAt(i) < 0)
        {
            changed += this->reassignPoint(i, kmeansState);
        }
        else
        {
            kmeansState->sqDistancesAt(i) = dist;
        }
    }

    return changed;
}

template <typename precision, typename int_size>
int_size OMPPointReassigner<precision, int_size>::reassignPoints(KmeansState<precision, int_size>* const kmeansState)
{
    int_size changed = 0;

#pragma omp parallel for shared(kmeansState), schedule(static), reduction(+ : changed)
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        changed += this->reassignPoint(i, kmeansState);
    }

    return changed;
}

template <typename precision, typename int_size>
int_size OMPOptimizedPointReassigner<precision, int_size>::reassignPoints(
  KmeansState<precision, int_size>* const kmeansState)
{
    int_size changed = 0;

#pragma omp parallel for shared(kmeansState), schedule(static), reduction(+ : changed)
    for (int_size i = 0; i < kmeansState->dataSize(); ++i)
    {
        auto clusterIdx = kmeansState->clusteringAt(i);
        auto dist       = std::pow((*kmeansState)(kmeansState->dataAt(i), kmeansState->clustersAt(clusterIdx)), 2);
        if (dist > kmeansState->sqDistancesAt(i) || kmeansState->sqDistancesAt(i) < 0)
        {
            changed += this->reassignPoint(i, kmeansState);
        }
        else
        {
            kmeansState->sqDistancesAt(i) = dist;
        }
    }

    return changed;
}
}  // namespace HPKmeans