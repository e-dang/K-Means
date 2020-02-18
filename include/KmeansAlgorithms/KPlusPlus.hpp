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
    std::fill((*(this->ppSqDistances))->begin(), (*(this->ppSqDistances))->end(), -1);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int_size i = 1; i < (*(this->ppClusters))->rows(); ++i)
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
      getRandDouble01() * std::accumulate((*(this->ppSqDistances))->begin(), (*(this->ppSqDistances))->end(), 0.0);
    int_size dataIdx = this->pSelector->select((*(this->ppSqDistances)), randSumFrac);
    (*(this->ppClusters))->push_back(this->pData->at(dataIdx));
}

template <typename precision, typename int_size>
void SharedMemoryKPlusPlus<precision, int_size>::findAndUpdateClosestClusters()
{
    this->pUpdater->findAndUpdateClosestClusters(this->pKmeansData);
}

template <typename precision, typename int_size>
void MPIKPlusPlus<precision, int_size>::weightedClusterSelection()
{
    int_size dataIdx;
    if (*(this->pRank) == 0)
    {
        precision randSumFrac = getRandDouble01MPI() * std::accumulate((*(this->ppSqDistances))->begin(),
                                                                       (*(this->ppSqDistances))->end(), 0.0);
        dataIdx               = this->pSelector->select(*(this->ppSqDistances), randSumFrac);
    }

    MPI_Bcast(&dataIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // find which rank holds the selected datapoint
    int_size rank = 0, lengthSum = 0;
    for (const auto& val : *(this->pLengths))
    {
        lengthSum += val;
        if (lengthSum > dataIdx)
            break;

        ++rank;
    }

    if (*(this->pRank) == rank)
    {
        (*(this->ppClusters))->push_back(this->pData->at(dataIdx - this->pDisplacements->at(*(this->pRank))));
    }
    else
    {
        (*(this->ppClusters))->reserve(1);
    }

    MPI_Bcast((*(this->ppClustering))->data(), (*(this->ppClustering))->size(), MPI_INT, rank, MPI_COMM_WORLD);
    MPI_Bcast((*(this->ppClusters))->data(), (*(this->ppClusters))->elements(), mpi_type_t, rank, MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
void MPIKPlusPlus<precision, int_size>::findAndUpdateClosestClusters()
{
    this->pUpdater->findAndUpdateClosestClusters(this->pKmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, this->pLengths->at(*(this->pRank)), MPI_INT, (*(this->ppClustering))->data(),
                   this->pLengths->data(), this->pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, this->pLengths->at(*(this->pRank)), mpi_type_t, (*(this->ppSqDistances))->data(),
                   this->pLengths->data(), this->pDisplacements->data(), mpi_type_t, MPI_COMM_WORLD);
}
}  // namespace HPKmeans