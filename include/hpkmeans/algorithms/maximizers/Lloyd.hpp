#pragma once

#include <mpi.h>

#include <hpkmeans/algorithms/maximizers/interface.hpp>
#include <hpkmeans/algorithms/strategies/averager.hpp>
#include <hpkmeans/algorithms/strategies/point_reassigner.hpp>
#include <hpkmeans/utils/mpi_class.hpp>

namespace HPKmeans
{
/**
 * @brief Implementation of a Kmeans maximization algorithm. Given a set of initialized clusters, this class will
 *        optimize the clusters using Lloyd's algorithm.
 */
template <typename precision, typename int_size>
class TemplateLloyd : public IKmeansMaximizer<precision, int_size>
{
protected:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;

    std::unique_ptr<AbstractPointReassigner<precision, int_size>> p_PointReassigner;
    std::unique_ptr<AbstractWeightedAverager<precision, int_size>> p_Averager;

public:
    TemplateLloyd(AbstractPointReassigner<precision, int_size>* pointReassigner,
                  AbstractWeightedAverager<precision, int_size>* averager) :
        p_PointReassigner(pointReassigner), p_Averager(averager)
    {
    }

    virtual ~TemplateLloyd() = default;

    /**
     * @brief Top level function for running Lloyd's algorithm on a set of pre-initialized clusters.
     */
    void maximize() final;

protected:
    /**
     * @brief Helper function that updates clusters based on the center of mass of the points assigned to it.
     */
    virtual void calcClusterSums() = 0;

    virtual void normalizeClusterSums() = 0;

    /**
     * @brief Helper function that checks if each point's closest cluster has changed after the clusters have been
     *        updated in the call to updateClusters(), and updates the clustering data if necessary. This function also
     *        keeps track of the number of datapoints that have changed cluster assignments and returns this value.

     * @return int - The number of datapoints whose cluster assignment has changed in the current iteration.
     */
    virtual int_size reassignPoints() = 0;
};

template <typename precision, typename int_size>
class SharedMemoryLloyd : public TemplateLloyd<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;

public:
    SharedMemoryLloyd(AbstractPointReassigner<precision, int_size>* pointReassigner,
                      AbstractWeightedAverager<precision, int_size>* averager) :
        TemplateLloyd<precision, int_size>(pointReassigner, averager)
    {
    }

    ~SharedMemoryLloyd() = default;

protected:
    void calcClusterSums() override;

    void normalizeClusterSums() override;

    int_size reassignPoints() override;
};

template <typename precision, typename int_size>
class MPILloyd : public TemplateLloyd<precision, int_size>, public MPIClass<precision, int_size>
{
private:
    using AbstractKmeansAlgorithm<precision, int_size>::p_KmeansState;
    using MPIClass<precision, int_size>::mpi_precision;
    using MPIClass<precision, int_size>::mpi_int_size;

public:
    MPILloyd(AbstractPointReassigner<precision, int_size>* pointReassigner,
             AbstractWeightedAverager<precision, int_size>* averager) :
        TemplateLloyd<precision, int_size>(pointReassigner, averager)
    {
    }

    ~MPILloyd() = default;

protected:
    void calcClusterSums() override;

    void normalizeClusterSums() override;

    int_size reassignPoints() override;
};

template <typename precision, typename int_size>
void TemplateLloyd<precision, int_size>::maximize()
{
    int_size changed, minNumChanged = p_KmeansState->totalNumData() * this->MIN_PERCENT_CHANGED;

    do
    {
        p_KmeansState->clustersFill(0.0);

        calcClusterSums();

        normalizeClusterSums();

        changed = reassignPoints();

    } while (changed > minNumChanged);
}

template <typename precision, typename int_size>
void SharedMemoryLloyd<precision, int_size>::calcClusterSums()
{
    this->p_Averager->calculateSum(p_KmeansState->data(), p_KmeansState->clusters(), p_KmeansState->clustering(),
                                   p_KmeansState->weights());
}

template <typename precision, typename int_size>
void SharedMemoryLloyd<precision, int_size>::normalizeClusterSums()
{
    this->p_Averager->normalizeSum(p_KmeansState->clusters(), p_KmeansState->clusterWeights());
}

template <typename precision, typename int_size>
int_size SharedMemoryLloyd<precision, int_size>::reassignPoints()
{
    return this->p_PointReassigner->reassignPoints(p_KmeansState);
}

template <typename precision, typename int_size>
void MPILloyd<precision, int_size>::calcClusterSums()
{
    this->p_Averager->calculateSum(p_KmeansState->data(), p_KmeansState->clusters(), p_KmeansState->clustering(),
                                   p_KmeansState->weights(), p_KmeansState->myDisplacement());

    MPI_Allreduce(MPI_IN_PLACE, p_KmeansState->clustersData(), p_KmeansState->clustersElements(), mpi_precision,
                  MPI_SUM, MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
void MPILloyd<precision, int_size>::normalizeClusterSums()
{
    std::vector<precision> copyWeights(p_KmeansState->dataSize());
    MPI_Reduce(p_KmeansState->clusterWeightsData(), copyWeights.data(), copyWeights.size(), mpi_precision, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (p_KmeansState->rank() == 0)
    {
        this->p_Averager->normalizeSum(p_KmeansState->clusters(), &copyWeights);
    }

    MPI_Bcast(p_KmeansState->clustersData(), p_KmeansState->clustersElements(), mpi_precision, 0, MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
int_size MPILloyd<precision, int_size>::reassignPoints()
{
    int_size changed = this->p_PointReassigner->reassignPoints(p_KmeansState);

    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, mpi_int_size, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, p_KmeansState->myLength(), mpi_int_size, p_KmeansState->clusteringData(),
                   p_KmeansState->lengthsData(), p_KmeansState->displacementsData(), mpi_int_size, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, p_KmeansState->myLength(), mpi_precision, p_KmeansState->sqDistancesData(),
                   p_KmeansState->lengthsData(), p_KmeansState->displacementsData(), mpi_precision, MPI_COMM_WORLD);

    return changed;
}
}  // namespace HPKmeans