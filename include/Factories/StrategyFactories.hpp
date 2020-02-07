#pragma once

#include "Averager.hpp"
#include "ClosestClusterFinder.hpp"
#include "ClosestClusterUpdater.hpp"
#include "CoresetClusteringFinisher.hpp"
#include "CoresetDistributionCalculator.hpp"
#include "DataClasses.hpp"
#include "DistanceSumCalculator.hpp"
#include "KmeansDataCreator.hpp"
#include "PointReassigner.hpp"
#include "RandomSelector.hpp"

class AbstractStrategyFactory
{
public:
    AbstractStrategyFactory(){};

    virtual ~AbstractStrategyFactory(){};

    IClosestClusterFinder* createClosestClusterFinder(Variant variant);

    AbstractClosestClusterUpdater* createClosestClusterUpdater(Variant variant);

    AbstractPointReassigner* createPointReassigner(Variant variant);

    IMultiWeightedRandomSelector* createMultiWeightedRandomSelector() { return new MultiWeightedRandomSelector(); };

    IWeightedRandomSelector* createWeightedRandomSelector() { return new SingleWeightedRandomSelector(); };

    AbstractClusteringDataUpdater* createCoresetClusteringDataUpdater() { return new CoresetClusteringDataUpdater(); }

    virtual AbstractClusteringDataUpdater* createClusteringDataUpdater() = 0;

    virtual AbstractWeightedAverager* createWeightedAverager() = 0;

    virtual AbstractAverager* createVectorAverager() = 0;

    virtual IDistanceSumCalculator* createDistanceSumCalculator() = 0;

    virtual ICoresetDistributionCalculator* createCoresetDistributionCalculator() = 0;

    virtual IKmeansDataCreator* createKmeansDataCreator() = 0;

    virtual AbstractCoresetClusteringFinisher* createCoresetClusteringFinisher() = 0;

    virtual AbstractPointReassigner* createRegPointReassigner() = 0;

    virtual AbstractPointReassigner* createOptPointReassigner() = 0;
};

class SerialStrategyFactory : public AbstractStrategyFactory
{
public:
    SerialStrategyFactory(){};

    ~SerialStrategyFactory(){};

    AbstractClusteringDataUpdater* createClusteringDataUpdater() override { return new ClusteringDataUpdater(); }

    AbstractPointReassigner* createRegPointReassigner() override
    {
        return new SerialPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner* createOptPointReassigner() override
    {
        return new SerialOptimizedPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager* createWeightedAverager() override { return new SerialWeightedMultiVectorAverager(); }

    AbstractAverager* createVectorAverager() override { return new SerialVectorAverager(); }

    IDistanceSumCalculator* createDistanceSumCalculator() override { return new SerialDistanceSumCalculator(); }

    ICoresetDistributionCalculator* createCoresetDistributionCalculator() override
    {
        return new SerialCoresetDistributionCalculator();
    }

    IKmeansDataCreator* createKmeansDataCreator() override { return new SharedMemoryKmeansDataCreator(); }

    AbstractCoresetClusteringFinisher* createCoresetClusteringFinisher() override
    {
        return new SharedMemoryCoresetClusteringFinisher(createClosestClusterUpdater(SpecificCoreset));
    }
};

class OMPStrategyFactory : public AbstractStrategyFactory
{
public:
    OMPStrategyFactory(){};

    ~OMPStrategyFactory(){};

    AbstractClusteringDataUpdater* createClusteringDataUpdater() override { return new AtomicClusteringDataUpdater(); }

    AbstractPointReassigner* createRegPointReassigner() override
    {
        return new OMPPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner* createOptPointReassigner() override
    {
        return new OMPOptimizedPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager* createWeightedAverager() override { return new OMPWeightedMultiVectorAverager(); }

    AbstractAverager* createVectorAverager() override { return new OMPVectorAverager(); }

    IDistanceSumCalculator* createDistanceSumCalculator() override { return new OMPDistanceSumCalculator(); }

    ICoresetDistributionCalculator* createCoresetDistributionCalculator() override
    {
        return new OMPCoresetDistributionCalculator();
    }

    IKmeansDataCreator* createKmeansDataCreator() override { return new SharedMemoryKmeansDataCreator(); }

    AbstractCoresetClusteringFinisher* createCoresetClusteringFinisher() override
    {
        return new SharedMemoryCoresetClusteringFinisher(createClosestClusterUpdater(SpecificCoreset));
    }
};

class MPIStrategyFactory : public AbstractStrategyFactory
{
public:
    MPIStrategyFactory(){};

    ~MPIStrategyFactory(){};

    AbstractClusteringDataUpdater* createClusteringDataUpdater() override
    {
        return new DistributedClusteringDataUpdater();
    }

    AbstractPointReassigner* createRegPointReassigner() override
    {
        return new SerialPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner* createOptPointReassigner() override
    {
        return new SerialOptimizedPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager* createWeightedAverager() override { return new SerialWeightedMultiVectorAverager(); }

    AbstractAverager* createVectorAverager() override { return new SerialVectorAverager(); }

    IDistanceSumCalculator* createDistanceSumCalculator() override { return new SerialDistanceSumCalculator(); }

    ICoresetDistributionCalculator* createCoresetDistributionCalculator() override { return nullptr; }

    IKmeansDataCreator* createKmeansDataCreator() override { return new MPIKmeansDataCreator(); }

    AbstractCoresetClusteringFinisher* createCoresetClusteringFinisher() override
    {
        return new MPICoresetClusteringFinisher(createClosestClusterUpdater(SpecificCoreset));
    }
};

class HybridStrategyFactory : public AbstractStrategyFactory
{
public:
    HybridStrategyFactory(){};

    ~HybridStrategyFactory(){};

    AbstractClusteringDataUpdater* createClusteringDataUpdater() override
    {
        return new AtomicDistributedClusteringDataUpdater();
    }

    AbstractPointReassigner* createRegPointReassigner() override
    {
        return new OMPPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner* createOptPointReassigner() override
    {
        return new OMPOptimizedPointReassigner(createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager* createWeightedAverager() override { return new OMPWeightedMultiVectorAverager(); }

    AbstractAverager* createVectorAverager() override { return new OMPVectorAverager(); }

    IDistanceSumCalculator* createDistanceSumCalculator() override { return new OMPDistanceSumCalculator(); }

    ICoresetDistributionCalculator* createCoresetDistributionCalculator() override { return nullptr; }

    IKmeansDataCreator* createKmeansDataCreator() override { return new MPIKmeansDataCreator(); }

    AbstractCoresetClusteringFinisher* createCoresetClusteringFinisher() override
    {
        return new MPICoresetClusteringFinisher(createClosestClusterUpdater(SpecificCoreset));
    }
};