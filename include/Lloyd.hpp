#pragma once

#include "Averager.hpp"
#include "KmeansAlgorithms.hpp"
#include "PointReassigner.hpp"

/**
 * @brief Implementation of a Kmeans maximization algorithm. Given a set of initialized clusters, this class will
 *        optimize the clusters using Lloyd's algorithm.
 */
class TemplateLloyd : public AbstractKmeansMaximizer
{
protected:
    std::unique_ptr<AbstractWeightedAverager> pAverager;

public:
    TemplateLloyd(AbstractPointReassigner* pointReassigner, AbstractWeightedAverager* averager) :
        AbstractKmeansMaximizer(pointReassigner), pAverager(averager){};

    virtual ~TemplateLloyd(){};

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

     * @return unsigned int - The number of datapoints whose cluster assignment has changed in the current iteration.
     */
    virtual unsigned int reassignPoints() = 0;
};

class SharedMemoryLloyd : public TemplateLloyd
{
public:
    SharedMemoryLloyd(AbstractPointReassigner* pointReassigner, AbstractWeightedAverager* averager) :
        TemplateLloyd(pointReassigner, averager){};

    ~SharedMemoryLloyd(){};

protected:
    void calcClusterSums() override;

    void averageClusterSums() override;

    unsigned int reassignPoints() override;
};

class MPILloyd : public TemplateLloyd
{
public:
    MPILloyd(AbstractPointReassigner* pointReassigner, AbstractWeightedAverager* averager) :
        TemplateLloyd(pointReassigner, averager){};

    ~MPILloyd(){};

protected:
    void calcClusterSums() override;

    void averageClusterSums() override;

    unsigned int reassignPoints() override;
};