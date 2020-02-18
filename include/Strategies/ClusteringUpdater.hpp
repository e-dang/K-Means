#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractClusteringDataUpdater
{
public:
    virtual ~AbstractClusteringDataUpdater() = default;

    void update(const int_size& dataIdx, const ClosestCluster<precision, int_size>& closestCluster,
                KmeansData<precision, int_size>* const kmeansData);

    virtual void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment,
                                      const int_size& newAssignment,
                                      KmeansData<precision, int_size>* const kmeansData) = 0;
};

template <typename precision, typename int_size>
class ClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~ClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
class AtomicClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~AtomicClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
class DistributedClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~DistributedClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
class AtomicDistributedClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~AtomicDistributedClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansData<precision, int_size>* const kmeansData);
};

template <typename precision, typename int_size>
class CoresetClusteringDataUpdater : public AbstractClusteringDataUpdater<precision, int_size>
{
public:
    ~CoresetClusteringDataUpdater() = default;

    void updateClusterWeights(const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
                              KmeansData<precision, int_size>* const kmeansData) override;
};

template <typename precision, typename int_size>
void AbstractClusteringDataUpdater<precision, int_size>::update(
  const int_size& dataIdx, const ClosestCluster<precision, int_size>& closestCluster,
  KmeansData<precision, int_size>* const kmeansData)
{
    if (kmeansData->sqDistancesAt(dataIdx) > closestCluster.distance || kmeansData->sqDistancesAt(dataIdx) < 0)
    {
        int_size& clusterAssignment = kmeansData->clusteringAt(dataIdx);
        if (clusterAssignment != closestCluster.clusterIdx)
        {
            updateClusterWeights(dataIdx, clusterAssignment, closestCluster.clusterIdx, kmeansData);
            clusterAssignment = closestCluster.clusterIdx;
        }
        kmeansData->sqDistancesAt(dataIdx) = closestCluster.distance;
    }
}

template <typename precision, typename int_size>
void ClusteringDataUpdater<precision, int_size>::updateClusterWeights(const int_size& dataIdx,
                                                                      const int_size& prevAssignment,
                                                                      const int_size& newAssignment,
                                                                      KmeansData<precision, int_size>* const kmeansData)
{
    precision weight = kmeansData->weights->at(dataIdx);
    if (prevAssignment >= 0 && kmeansData->clusterWeightsAt(prevAssignment) > 0)
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void AtomicClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansData<precision, int_size>* const kmeansData)
{
    precision weight = kmeansData->weights->at(dataIdx);
    if (prevAssignment >= 0 && kmeansData->clusterWeightsAt(prevAssignment) > 0)
#pragma omp atomic
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
#pragma omp atomic
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void DistributedClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansData<precision, int_size>* const kmeansData)
{
    precision weight = kmeansData->weights->at(dataIdx);
    if (prevAssignment >= 0)
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void AtomicDistributedClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansData<precision, int_size>* const kmeansData)
{
    precision weight = kmeansData->weights->at(dataIdx);
    if (prevAssignment >= 0)
#pragma omp atomic
        kmeansData->clusterWeightsAt(prevAssignment) -= weight;
#pragma omp atomic
    kmeansData->clusterWeightsAt(newAssignment) += weight;
}

template <typename precision, typename int_size>
void CoresetClusteringDataUpdater<precision, int_size>::updateClusterWeights(
  const int_size& dataIdx, const int_size& prevAssignment, const int_size& newAssignment,
  KmeansData<precision, int_size>* const kmeansData)
{
    // no operations
}
}  // namespace HPKmeans