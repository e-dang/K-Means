#pragma once

#include "AbstractKmeansAlgorithms.hpp"
#include "RandomSelector.hpp"

/**
 * @brief Implementation of a Kmeans++ initialization aglorithm. Selects datapoints to be new clusters at random
 *        weighted by the square distance between the point and its nearest cluster. Thus farther points have a higher
 *        probability of being selected.
 */
class TemplateKPlusPlus : public AbstractKmeansInitializer
{
protected:
    std::unique_ptr<IWeightedRandomSelector> pSelector;

public:
    /**
     * @brief Construct a new TemplateKPlusPlus object
     *
     * @param updater
     * @param selector
     */
    TemplateKPlusPlus(AbstractClosestClusterUpdater* updater, IWeightedRandomSelector* selector) :
        AbstractKmeansInitializer(updater), pSelector(selector){};

    /**
     * @brief Destroy the Serial KPlusPlus object
     */
    virtual ~TemplateKPlusPlus(){};

    /**
     * @brief Template function that initializes the clusters.
     */
    void initialize();

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

class SharedMemoryKPlusPlus : public TemplateKPlusPlus
{
public:
    SharedMemoryKPlusPlus(AbstractClosestClusterUpdater* updater, IWeightedRandomSelector* selector) :
        TemplateKPlusPlus(updater, selector){};

    ~SharedMemoryKPlusPlus(){};

    void weightedClusterSelection() override;

    void findAndUpdateClosestClusters() override;
};

class MPIKPlusPlus : public TemplateKPlusPlus
{
public:
    MPIKPlusPlus(AbstractClosestClusterUpdater* updater, IWeightedRandomSelector* selector) :
        TemplateKPlusPlus(updater, selector){};

    ~MPIKPlusPlus(){};

protected:
    void weightedClusterSelection() override;

    void findAndUpdateClosestClusters() override;
};