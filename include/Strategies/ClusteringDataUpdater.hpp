#pragma once

#include "Containers/KmeansState.hpp"

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractClusteringDataUpdater
{
public:
    virtual ~AbstractClusteringDataUpdater() = default;

    void update(const int_size& dataIdx, const ClosestCluster<precision, int_size>& closestCluster,
                KmeansState<precision, int_size>* const kmeansState);

    virtual void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment,
                                      const int_size& newAssignment,
                                      KmeansState<precision, int_size>* const kmeansState) = 0;
};

template <typename precision, typename int_size>
class ClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~ClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class AtomicClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~AtomicClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class DistributedClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~DistributedClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
class AtomicDistributedClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~AtomicDistributedClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansState<precision, int_size>* const kmeansState);
};

template <typename precision, typename int_size>
class CoresetClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~CoresetClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansState<precision, int_size>* const kmeansState) override;
};

template <typename precision, typename int_size>
void AbstractClusteringDataUpdater<precision, int_size>::update(
  const int_size& dataIdx, const ClosestCluster<precision, int_size>& closestCluster,
  KmeansState<precision, int_size>* const kmeansState)
{
    if (kmeansState->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansState->sqDistancesAt(dataIdx) < 0)
    {
        int_size& clusterAssignment = kmeansState->clusteringAt(dataIdx);
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            updateClusterWeights(dataIdx, clusterAssignment, closestCluster.clusterIdx, kmeansState);
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansState->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}

template <typename precision, typename int_size>
void ClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansState<precision, int_size>* const kmeansState)
{
    precision weight = kmeansState->weightsAt(dataIdx);
    if (prevAssignment >= 0 && kmeansState->clusterWeightsAt(prevAssignment) > 0)
        kmeansState->clusterWeightsAt(prevAssignment) -= weight;
    kmeansState->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void AtomicClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansState<precision, int_size>* const kmeansState)
{
    precision weight = kmeansState->weightsAt(dataIdx);
    if (prevAssignment >= 0 && kmeansState->clusterWeightsAt(prevAssignment) > 0)
#pragma omp atomic
        kmeansState->clusterWeightsAt(prevAssignment) -= weight;
#pragma omp atomic
    kmeansState->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void DistributedClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansState<precision, int_size>* const kmeansState)
{
    precision weight = kmeansState->weightsAt(dataIdx);
    if (prevAssignment >= 0)
        kmeansState->clusterWeightsAt(prevAssignment) -= weight;
    kmeansState->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void AtomicDistributedClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansState<precision, int_size>* const kmeansState)
{
    precision weight = kmeansState->weightsAt(dataIdx);
    if (prevAssignment >= 0)
#pragma omp atomic
        kmeansState->clusterWeightsAt(prevAssignment) -= weight;
#pragma omp atomic
    kmeansState->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void CoresetClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansState<precision, int_size>* const kmeansState)
{
    // no operations
}
}  // namespace HPKmeans