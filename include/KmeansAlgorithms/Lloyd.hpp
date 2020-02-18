#pragma once

#include "KmeansAlgorithms/KmeansAlgorithms.hpp"
#include "Strategies/Averager.hpp"
#include "Strategies/PointReassigner.hpp"
#include "mpi.h"
namespace HPKmeans
{
/**
 * @brief Implementation of a Kmeans maximization algorithm. Given a set of initialized clusters, this class will
 *        optimize the clusters using Lloyd's algorithm.
 */
template <typename precision, typename int_size>
class TemplateLloyd : public AbstractKmeansMaximizer<precision, int_size>
{
protected:
    std::unique_ptr<AbstractWeightedAverager<precision, int_size>> pAverager;

public:
    TemplateLloyd(AbstractPointReassigner<precision, int_size>* pointReassigner,
                  AbstractWeightedAverager<precision, int_size>* averager) :
        AbstractKmeansMaximizer<precision, int_size>(pointReassigner), pAverager(averager)
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

    virtual void averageClusterSums() = 0;

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
public:
    SharedMemoryLloyd(AbstractPointReassigner<precision, int_size>* pointReassigner,
                      AbstractWeightedAverager<precision, int_size>* averager) :
        TemplateLloyd<precision, int_size>(pointReassigner, averager)
    {
    }

    ~SharedMemoryLloyd() = default;

protected:
    void calcClusterSums() override;

    void averageClusterSums() override;

    int_size reassignPoints() override;
};

template <typename precision, typename int_size>
class MPILloyd : public TemplateLloyd<precision, int_size>
{
public:
    MPILloyd(AbstractPointReassigner<precision, int_size>* pointReassigner,
             AbstractWeightedAverager<precision, int_size>* averager) :
        TemplateLloyd<precision, int_size>(pointReassigner, averager)
    {
    }

    ~MPILloyd() = default;

protected:
    void calcClusterSums() override;

    void averageClusterSums() override;

    int_size reassignPoints() override;
};

template <typename precision, typename int_size>
void TemplateLloyd<precision, int_size>::maximize()
{
    int_size changed, minNumChanged = (*(this->pTotalNumData) * this->MIN_PERCENT_CHANGED);

    do
    {
        (*(this->ppClusters))->fill(0);

        calcClusterSums();

        averageClusterSums();

        changed = reassignPoints();

    } while (changed > minNumChanged);
}

template <typename precision, typename int_size>
void SharedMemoryLloyd<precision, int_size>::calcClusterSums()
{
    this->pAverager->calculateSum(this->pData, *(this->ppClusters), *(this->ppClustering), this->pWeights);
}

template <typename precision, typename int_size>
void SharedMemoryLloyd<precision, int_size>::averageClusterSums()
{
    this->pAverager->normalizeSum(*(this->ppClusters), *(this->ppClusterWeights));
}

template <typename precision, typename int_size>
int_size SharedMemoryLloyd<precision, int_size>::reassignPoints()
{
    return this->pPointReassigner->reassignPoints(this->pKmeansData);
}

template <typename precision, typename int_size>
void MPILloyd<precision, int_size>::calcClusterSums()
{
    this->pAverager->calculateSum(this->pData, *(this->ppClusters), *(this->ppClustering), this->pWeights,
                                  this->pDisplacements->at(*(this->pRank)));

    MPI_Allreduce(MPI_IN_PLACE, (*(this->ppClusters))->data(), (*(this->ppClusters))->elements(), mpi_type_t, MPI_SUM,
                  MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
void MPILloyd<precision, int_size>::averageClusterSums()
{
    std::vector<precision> copyWeights((*(this->ppClusterWeights))->size());
    MPI_Reduce((*(this->ppClusterWeights))->data(), copyWeights.data(), copyWeights.size(), mpi_type_t, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (*(this->pRank) == 0)
    {
        this->pAverager->normalizeSum(*(this->ppClusters), &copyWeights);
    }

    MPI_Bcast((*(this->ppClusters))->data(), (*(this->ppClusters))->elements(), mpi_type_t, 0, MPI_COMM_WORLD);
}

template <typename precision, typename int_size>
int_size MPILloyd<precision, int_size>::reassignPoints()
{
    int_size changed = this->pPointReassigner->reassignPoints(this->pKmeansData);

    MPI_Allgatherv(MPI_IN_PLACE, this->pLengths->at(*(this->pRank)), MPI_INT, (*(this->ppClustering))->data(),
                   this->pLengths->data(), this->pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, this->pLengths->at(*(this->pRank)), mpi_type_t, (*this->ppSqDistances)->data(),
                   this->pLengths->data(), this->pDisplacements->data(), mpi_type_t, MPI_COMM_WORLD);

    return changed;
}
}  // namespace HPKmeans