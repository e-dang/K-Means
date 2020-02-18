#pragma once

#include <omp.h>

#include "KmeansAlgorithms/KmeansAlgorithms.hpp"
#include "Strategies/RandomSelector.hpp"
#include "Utils/Utils.hpp"
#include "mpi.h"
namespace HPKmeans
{
/**
 * @brief Implementation of a Kmeans++ initialization aglorithm. Selects datapoints to be new clusters at random
 *        weighted by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 */
template <typename precision, typename int_size>
class TemplateKPlusPlus : public AbstractKmeansInitializer<precision, int_size>
{
protected:
    using AbstractKmeansAlgorithm<precision, int_size>::pKmeansData;

protected:
    std::unique_ptr<IWeightedRandomSelector<precision, int_size>> pSelector;

public:
    /**
     * @brief Construct a new TemplateKPlusPlus object
     *
     * @param updater
     * @param selector
     */
    TemplateKPlusPlus(AbstractClosestClusterUpdater<precision, int_size>* updater,
                      IWeightedRandomSelector<precision, int_size>* selector) :
        AbstractKmeansInitializer<precision, int_size>(updater), pSelector(selector)
    {
    }

    /**
     * @brief Destroy the Serial KPlusPlus object
     */
    virtual ~TemplateKPlusPlus() = default;

    /**
     * @brief Template function that initializes the clusters.
     */
    void initialize() final;

protected:
    /**
     * @brief Helper function that selects a datapoint to be a new cluster center with a probability proportional to the
     *        square of the distance to its current closest cluster.
     */
    virtual void weightedClusterSelection() = 0;

    /**
     * @brief Helper function that wraps the functionality of findClosestCluster() and updateClustering() in order to
     *        find the closest cluster for each datapoint and update the clustering assignments.
     */
    virtual void findAndUpdateClosestClusters() = 0;
};

template <typename precision, typename int_size>
class SharedMemoryKPlusPlus : public TemplateKPlusPlus<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::pKmeansData;

public:
    SharedMemoryKPlusPlus(AbstractClosestClusterUpdater<precision, int_size>* updater,
                          IWeightedRandomSelector<precision, int_size>* selector) :
        TemplateKPlusPlus<precision, int_size>(updater, selector)
    {
    }

    ~SharedMemoryKPlusPlus() = default;

protected:
    void weightedClusterSelection() override;

    void findAndUpdateClosestClusters() override;
};

template <typename precision, typename int_size>
class MPIKPlusPlus : public TemplateKPlusPlus<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::pKmeansData;

public:
    MPIKPlusPlus(AbstractClosestClusterUpdater<precision, int_size>* updater,
                 IWeightedRandomSelector<precision, int_size>* selector) :
        TemplateKPlusPlus<precision, int_size>(updater, selector)
    {
    }

    ~MPIKPlusPlus() = default;

protected:
    void weightedClusterSelection() override;

    void findAndUpdateClosestClusters() override;
};

template <typename precision, typename int_size>
void TemplateKPlusPlus<precision, int_size>::initialize()
{
    // initialize first cluster uniformly at random. Thus distances should be filled with same number i.e. 1
    weightedClusterSelection();

    // change fill distances vector with -1 so values aren't confused with actual distances
    std::fill(pKmeansData->sqDistancesBegin(), pKmeansData->sqDistancesEnd(), -1.0);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int_size i = 1; i < pKmeansData->clustersRows(); ++i)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestClusters();

        // select point to be next cluster center weighted by nearest distance squared
        weightedClusterSelection();
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestClusters();
}

template <typename precision, typename int_size>
void SharedMemoryKPlusPlus<precision, int_size>::weightedClusterSelection()
{
    precision randSumFrac =
      getRandDouble01() * std::accumulate(pKmeansData->sqDistancesBegin(), pKmeansData->sqDistancesEnd(), 0.0);
    int_size dataIdx = this->pSelector->select(pKmeansData->sqDistances(), randSumFrac);
    pKmeansData->clustersPushBack(pKmeansData->dataAt(dataIdx));
}

template <typename precision, typename int_size>
void SharedMemoryKPlusPlus<precision, int_size>::findAndUpdateClosestClusters()
{
    this->pUpdater->findAndUpdateClosestClusters(pKmeansData);
}

template <typename precision, typename int_size>
void MPIKPlusPlus<precision, int_size>::weightedClusterSelection()
{
    int_size dataIdx;
    if (pKmeansData->rank() == 0)
    {
        precision randSumFrac =
          getRandDouble01MPI() * std::accumulate(pKmeansData->sqDistancesBegin(), pKmeansData->sqDistancesEnd(), 0.0);
        dataIdx = this->pSelector->select(pKmeansData->sqDistances(), randSumFrac);
    }

    MPI_Bcast(&dataIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // find which rank holds the selected datapoint
    int_size rank = 0, lengthSum = 0;
    for (const auto& val : pKmeansData->lengths())
    {
        lengthSum += val;
        if (lengthSum > dataIdx)
            break;

        ++rank;
    }

    if (pKmeansData->rank() == rank)
    {
        pKmeansData->clustersPushBack(pKmeansData->dataAt(dataIdx - pKmeansData->myDisplacement()));
    }
    else
    {
        pKmeansData->clustersReserve(1);
    }

    MPI_Bcast(pKmeansData->clusteringData(), pKmeansData->clusteringSize(), MPI_INT, rank, MPI_COMM_WORLD);
    MPI_Bcast(pKmeansData->clustersData(), pKmeansData->clustersElements(), mpi_type_t, rank, MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
void MPIKPlusPlus<precision, int_size>::findAndUpdateClosestClusters()
{
    this->pUpdater->findAndUpdateClosestClusters(pKmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, pKmeansData->myLength(), MPI_INT, pKmeansData->clusteringData(),
                   pKmeansData->lengthsData(), pKmeansData->displacementsData(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pKmeansData->myLength(), mpi_type_t, pKmeansData->sqDistancesData(),
                   pKmeansData->lengthsData(), pKmeansData->displacementsData(), mpi_type_t, MPI_COMM_WORLD);
}
}  // namespace HPKmeans