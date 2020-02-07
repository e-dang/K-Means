#pragma once

#include "ClosestClusterUpdater.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"

class AbstractPointReassigner
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater> pUpdater;

public:
    AbstractPointReassigner(AbstractClosestClusterUpdater* updater) : pUpdater(updater){};

    virtual ~AbstractPointReassigner(){};

    int_fast32_t reassignPoint(const int_fast32_t& dataIdx, KmeansData* const kmeansData)
    {
        auto before = kmeansData->clusteringAt(dataIdx);

        pUpdater->findAndUpdateClosestCluster(dataIdx, kmeansData);

        if (before != kmeansData->clusteringAt(dataIdx))
        {
            return 1;
        }
        return 0;
    }

    virtual int_fast32_t reassignPoints(KmeansData* const kmeansData) = 0;
};

class SerialPointReassigner : public AbstractPointReassigner
{
public:
    SerialPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~SerialPointReassigner(){};

    int_fast32_t reassignPoints(KmeansData* const kmeansData) override
    {
        int_fast32_t changed = 0;
        for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            changed += reassignPoint(i, kmeansData);
        }

        return changed;
    }
};

class SerialOptimizedPointReassigner : public AbstractPointReassigner
{
public:
    SerialOptimizedPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~SerialOptimizedPointReassigner(){};

    int_fast32_t reassignPoints(KmeansData* const kmeansData) override
    {
        int_fast32_t changed = 0;
        auto numFeatures     = kmeansData->pData->getNumFeatures();

        for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            auto clusterIdx = kmeansData->clusteringAt(i);
            auto dist       = std::pow((*kmeansData->pDistanceFunc)(kmeansData->pData->at(i),
                                                              kmeansData->pClusters->at(clusterIdx), numFeatures),
                                 2);
            if (dist > kmeansData->sqDistancesAt(i) || kmeansData->sqDistancesAt(i) < 0)
            {
                changed += reassignPoint(i, kmeansData);
            }
            else
            {
                kmeansData->sqDistancesAt(i) = dist;
            }
        }

        return changed;
    }
};

class OMPPointReassigner : public AbstractPointReassigner
{
public:
    OMPPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~OMPPointReassigner(){};

    int_fast32_t reassignPoints(KmeansData* const kmeansData) override
    {
        int_fast32_t changed = 0;

#pragma omp parallel for shared(kmeansData), schedule(static), reduction(+ : changed)
        for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            changed += reassignPoint(i, kmeansData);
        }

        return changed;
    }
};

class OMPOptimizedPointReassigner : public AbstractPointReassigner
{
public:
    OMPOptimizedPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~OMPOptimizedPointReassigner(){};

    int_fast32_t reassignPoints(KmeansData* const kmeansData) override
    {
        int_fast32_t changed = 0;
        auto numFeatures     = kmeansData->pData->getNumFeatures();

#pragma omp parallel for shared(kmeansData, numFeatures), schedule(static), reduction(+ : changed)
        for (int_fast32_t i = 0; i < kmeansData->pData->getNumData(); i++)
        {
            auto clusterIdx = kmeansData->clusteringAt(i);
            auto dist       = std::pow((*kmeansData->pDistanceFunc)(kmeansData->pData->at(i),
                                                              kmeansData->pClusters->at(clusterIdx), numFeatures),
                                 2);
            if (dist > kmeansData->sqDistancesAt(i) || kmeansData->sqDistancesAt(i) < 0)
            {
                changed += reassignPoint(i, kmeansData);
            }
            else
            {
                kmeansData->sqDistancesAt(i) = dist;
            }
        }

        return changed;
    }
};