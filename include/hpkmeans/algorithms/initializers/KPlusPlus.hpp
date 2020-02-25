#pragma once

#include <mpi.h>
#include <omp.h>

#include <hpkmeans/algorithms/initializers/interface.hpp>
#include <hpkmeans/algorithms/strategies/closest_cluster_updater.hpp>
#include <hpkmeans/algorithms/strategies/random_selector.hpp>
#include <hpkmeans/utils/Utils.hpp>
#include <hpkmeans/utils/mpi_class.hpp>

namespace HPKmeans
{
/**
 * @brief Implementation of a Kmeans++ initialization aglorithm. Selects datapoints to be new clusters at random
 *        weighted by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 */
template <typename precision, typename int_size>
class TemplateKPlusPlus : public IKmeansInitializer<precision, int_size>
{
protected:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;

    std::unique_ptr<AbstractClosestClusterUpdater<precision, int_size>> p_Updater;
    std::unique_ptr<IWeightedRandomSelector<precision, int_size>> p_Selector;

public:
    /**
     * @brief Construct a new TemplateKPlusPlus object
     *
     * @param updater
     * @param selector
     */
    TemplateKPlusPlus(AbstractClosestClusterUpdater<precision, int_size>* updater,
                      IWeightedRandomSelector<precision, int_size>* selector) :
        p_Updater(updater), p_Selector(selector)
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
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;

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
class MPIKPlusPlus : public TemplateKPlusPlus<precision, int_size>, public MPIClass<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;
    using MPIClass<precision, int_size>::mpi_precision;
    using MPIClass<precision, int_size>::mpi_int_size;

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
    std::fill(p_KmeansState->sqDistancesBegin(), p_KmeansState->sqDistancesEnd(), -1.0);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int_size i = 1; i < p_KmeansState->clustersRows(); ++i)
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
      getRandDouble01() * std::accumulate(p_KmeansState->sqDistancesBegin(), p_KmeansState->sqDistancesEnd(), 0.0);
    int_size dataIdx = this->p_Selector->select(p_KmeansState->sqDistances(), randSumFrac);
    p_KmeansState->clustersPushBack(p_KmeansState->dataAt(dataIdx));
}

template <typename precision, typename int_size>
void SharedMemoryKPlusPlus<precision, int_size>::findAndUpdateClosestClusters()
{
    this->p_Updater->findAndUpdateClosestClusters(p_KmeansState);
}

template <typename precision, typename int_size>
void MPIKPlusPlus<precision, int_size>::weightedClusterSelection()
{
    int_size dataIdx;
    if (p_KmeansState->rank() == 0)
    {
        precision randSumFrac = getRandDouble01MPI() * std::accumulate(p_KmeansState->sqDistancesBegin(),
                                                                       p_KmeansState->sqDistancesEnd(), 0.0);
        dataIdx               = this->p_Selector->select(p_KmeansState->sqDistances(), randSumFrac);
    }

    MPI_Bcast(&dataIdx, 1, mpi_int_size, 0, MPI_COMM_WORLD);

    // find which rank holds the selected datapoint
    int_size rank = 0, lengthSum = 0;
    for (const auto& val : p_KmeansState->lengths())
    {
        lengthSum += val;
        if (lengthSum > dataIdx)
            break;

        ++rank;
    }

    if (p_KmeansState->rank() == rank)
    {
        p_KmeansState->clustersPushBack(p_KmeansState->dataAt(dataIdx - p_KmeansState->myDisplacement()));
    }
    else
    {
        p_KmeansState->clustersReserve(1);
    }

    MPI_Bcast(p_KmeansState->clusteringData(), p_KmeansState->clusteringSize(), mpi_int_size, rank, MPI_COMM_WORLD);
    MPI_Bcast(p_KmeansState->clustersData(), p_KmeansState->clustersElements(), mpi_precision, rank, MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
void MPIKPlusPlus<precision, int_size>::findAndUpdateClosestClusters()
{
    this->p_Updater->findAndUpdateClosestClusters(p_KmeansState);

    MPI_Allgatherv(MPI_IN_PLACE, p_KmeansState->myLength(), mpi_int_size, p_KmeansState->clusteringData(),
                   p_KmeansState->lengthsData(), p_KmeansState->displacementsData(), mpi_int_size, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, p_KmeansState->myLength(), mpi_precision, p_KmeansState->sqDistancesData(),
                   p_KmeansState->lengthsData(), p_KmeansState->displacementsData(), mpi_precision, MPI_COMM_WORLD);
}
}  // namespace HPKmeans