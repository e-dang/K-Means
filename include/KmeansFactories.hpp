#pragma once

#include "ClosestClusterUpdater.hpp"
#include "CoresetClusteringFinisher.hpp"
#include "CoresetCreator.hpp"
#include "CoresetDistributionCalculator.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "DistanceSumCalculator.hpp"
#include "KPlusPlus.hpp"
#include "Kmeans.hpp"
#include "KmeansAlgorithms.hpp"
#include "Lloyd.hpp"

enum Initializer
{
    KPP    = 1 << 0,
    OptKPP = 1 << 1
};

enum Maximizer
{
    Lloyd    = 1 << 2,
    OptLloyd = 1 << 3
};

enum CoresetCreator
{
    LWCoreset = 1 << 4,
    None      = 0
};

enum Parallelism
{
    Serial = 1 << 5,
    OMP    = 1 << 6,
    MPI    = 1 << 7,
    Hybrid = 1 << 8
};

enum Variant
{
    Reg,
    Opt,
    SpecificCoreset
};

class AbstractStrategyFactory
{
public:
    AbstractStrategyFactory(){};

    virtual ~AbstractStrategyFactory(){};

    virtual AbstractClusteringDataUpdater* createClusteringDataUpdater() = 0;

    virtual AbstractClosestClusterUpdater* createClosestClusterUpdater(Variant variant) = 0;

    virtual AbstractPointReassigner* createPointReassigner(Variant variant) = 0;

    virtual AbstractWeightedAverager* createWeightedAverager() = 0;

    virtual AbstractAverager* createVectorAverager() = 0;

    virtual IDistanceSumCalculator* createDistanceSumCalculator() = 0;

    virtual ICoresetDistributionCalculator* createCoresetDistributionCalculator() = 0;

    virtual IKmeansDataCreator* createKmeansDataCreator() = 0;

    virtual AbstractCoresetClusteringFinisher* createCoresetClusteringFinisher() = 0;

    IClosestClusterFinder* createClosestClusterFinder(Variant variant)
    {
        switch (variant)
        {
            case (Reg):
                return new ClosestClusterFinder();
            case (Opt):
                return new ClosestNewClusterFinder();
        }
    }

    IMultiWeightedRandomSelector* createMultiWeightedRandomSelector() { return new MultiWeightedRandomSelector(); };

    IWeightedRandomSelector* createWeightedRandomSelector() { return new SingleWeightedRandomSelector(); };

    AbstractClusteringDataUpdater* createCoresetClusteringDataUpdater() { return new CoresetClusteringDataUpdater(); }
};

class SerialStrategyFactory : public AbstractStrategyFactory
{
public:
    SerialStrategyFactory(){};

    ~SerialStrategyFactory(){};

    AbstractClusteringDataUpdater* createClusteringDataUpdater() override { return new ClusteringDataUpdater(); }

    AbstractClosestClusterUpdater* createClosestClusterUpdater(Variant variant) override
    {
        switch (variant)
        {
            case (SpecificCoreset):
                return new SerialClosestClusterUpdater(createClosestClusterFinder(Reg),
                                                       createCoresetClusteringDataUpdater());
            default:
                return new SerialClosestClusterUpdater(createClosestClusterFinder(variant),
                                                       createClusteringDataUpdater());
        }
    }

    AbstractPointReassigner* createPointReassigner(Variant variant) override
    {
        switch (variant)
        {
            case (Reg):
                return new SerialPointReassigner(createClosestClusterUpdater(Reg));
            case (Opt):
                return new SerialOptimizedPointReassigner(createClosestClusterUpdater(Reg));
        }
    }

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

    AbstractClosestClusterUpdater* createClosestClusterUpdater(Variant variant) override
    {
        switch (variant)
        {
            case (SpecificCoreset):
                return new OMPClosestClusterUpdater(createClosestClusterFinder(Reg),
                                                    createCoresetClusteringDataUpdater());
            default:
                return new OMPClosestClusterUpdater(createClosestClusterFinder(variant), createClusteringDataUpdater());
        }
    }

    AbstractPointReassigner* createPointReassigner(Variant variant) override
    {
        switch (variant)
        {
            case (Reg):
                return new OMPPointReassigner(createClosestClusterUpdater(Reg));
            case (Opt):
                return new OMPOptimizedPointReassigner(createClosestClusterUpdater(Reg));
        }
    }

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

    AbstractClosestClusterUpdater* createClosestClusterUpdater(Variant variant) override
    {
        switch (variant)
        {
            case (SpecificCoreset):
                return new SerialClosestClusterUpdater(createClosestClusterFinder(Reg),
                                                       createCoresetClusteringDataUpdater());
            default:
                return new SerialClosestClusterUpdater(createClosestClusterFinder(variant),
                                                       createClusteringDataUpdater());
        }
    }

    AbstractPointReassigner* createPointReassigner(Variant variant) override
    {
        switch (variant)
        {
            case (Reg):
                return new SerialPointReassigner(createClosestClusterUpdater(Reg));
            case (Opt):
                return new SerialOptimizedPointReassigner(createClosestClusterUpdater(Reg));
        }
    }

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

    AbstractClosestClusterUpdater* createClosestClusterUpdater(Variant variant) override
    {
        switch (variant)
        {
            case (SpecificCoreset):
                return new OMPClosestClusterUpdater(createClosestClusterFinder(Reg),
                                                    createCoresetClusteringDataUpdater());
            default:
                return new OMPClosestClusterUpdater(createClosestClusterFinder(variant), createClusteringDataUpdater());
        }
    }

    AbstractPointReassigner* createPointReassigner(Variant variant) override
    {
        switch (variant)
        {
            case (Reg):
                return new OMPPointReassigner(createClosestClusterUpdater(Reg));
            case (Opt):
                return new OMPOptimizedPointReassigner(createClosestClusterUpdater(Reg));
        }
    }

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

class AbstractKmeansAlgorithmFactory
{
protected:
    std::unique_ptr<AbstractStrategyFactory> pStratFactory;

public:
    AbstractKmeansAlgorithmFactory(AbstractStrategyFactory* stratFactory) : pStratFactory(stratFactory){};

    virtual ~AbstractKmeansAlgorithmFactory(){};

    AbstractKmeansInitializer* createInitializer(Initializer initializer)
    {
        switch (initializer)
        {
            case (KPP):
                return getKPP();
            case (OptKPP):
                return getOptKPP();
            default:
                throw std::runtime_error("Invalid initializer specified.");
        }
    }

    AbstractKmeansMaximizer* createMaximizer(Maximizer maximizer)
    {
        switch (maximizer)
        {
            case (Lloyd):
                return getLloyd();
            case (OptLloyd):
                return getOptLloyd();
            default:
                throw std::runtime_error("Invalid maximizer specified.");
        }
    }

    AbstractCoresetCreator* createCoresetCreator(CoresetCreator coresetCreator, const size_t& sampleSize,
                                                 std::shared_ptr<IDistanceFunctor> distanceFunc)
    {
        switch (coresetCreator)
        {
            case (LWCoreset):
                return getLWCoreset(sampleSize, distanceFunc);
            default:
                return nullptr;
        }
    }

protected:
    virtual AbstractKmeansInitializer* getKPP() = 0;

    virtual AbstractKmeansInitializer* getOptKPP() = 0;

    virtual AbstractKmeansMaximizer* getLloyd() = 0;

    virtual AbstractKmeansMaximizer* getOptLloyd() = 0;

    virtual AbstractCoresetCreator* getLWCoreset(const size_t& sampleSize,
                                                 std::shared_ptr<IDistanceFunctor> distanceFunc) = 0;
};

class SharedMemoryKmeansAlgorithmFactory : public AbstractKmeansAlgorithmFactory
{
public:
    SharedMemoryKmeansAlgorithmFactory(AbstractStrategyFactory* stratFactory) :
        AbstractKmeansAlgorithmFactory(stratFactory){};

    ~SharedMemoryKmeansAlgorithmFactory(){};

    AbstractKmeansInitializer* getKPP()
    {
        return new SharedMemoryKPlusPlus(pStratFactory->createClosestClusterUpdater(Reg),
                                         pStratFactory->createWeightedRandomSelector());
    }

    AbstractKmeansInitializer* getOptKPP()
    {
        return new SharedMemoryKPlusPlus(pStratFactory->createClosestClusterUpdater(Opt),
                                         pStratFactory->createWeightedRandomSelector());
    }

    AbstractKmeansMaximizer* getLloyd()
    {
        return new SharedMemoryLloyd(pStratFactory->createPointReassigner(Reg),
                                     pStratFactory->createWeightedAverager());
    }

    AbstractKmeansMaximizer* getOptLloyd()
    {
        return new SharedMemoryLloyd(pStratFactory->createPointReassigner(Opt),
                                     pStratFactory->createWeightedAverager());
    }

    AbstractCoresetCreator* getLWCoreset(const size_t& sampleSize, std::shared_ptr<IDistanceFunctor> distanceFunc)
    {
        return new SharedMemoryCoresetCreator(sampleSize, pStratFactory->createMultiWeightedRandomSelector(),
                                              pStratFactory->createVectorAverager(),
                                              pStratFactory->createDistanceSumCalculator(),
                                              pStratFactory->createCoresetDistributionCalculator(), distanceFunc);
    }
};

class MPIKmeansAlgorithmFactory : public AbstractKmeansAlgorithmFactory
{
public:
    MPIKmeansAlgorithmFactory(AbstractStrategyFactory* stratFactory) : AbstractKmeansAlgorithmFactory(stratFactory){};

    ~MPIKmeansAlgorithmFactory(){};

    AbstractKmeansInitializer* getKPP()
    {
        return new MPIKPlusPlus(pStratFactory->createClosestClusterUpdater(Reg),
                                pStratFactory->createWeightedRandomSelector());
    }

    AbstractKmeansInitializer* getOptKPP()
    {
        return new MPIKPlusPlus(pStratFactory->createClosestClusterUpdater(Opt),
                                pStratFactory->createWeightedRandomSelector());
    }

    AbstractKmeansMaximizer* getLloyd()
    {
        return new MPILloyd(pStratFactory->createPointReassigner(Reg), pStratFactory->createWeightedAverager());
    }

    AbstractKmeansMaximizer* getOptLloyd()
    {
        return new MPILloyd(pStratFactory->createPointReassigner(Opt), pStratFactory->createWeightedAverager());
    }

    AbstractCoresetCreator* getLWCoreset(const size_t& sampleSize, std::shared_ptr<IDistanceFunctor> distanceFunc)
    {
        return new MPICoresetCreator(sampleSize, pStratFactory->createMultiWeightedRandomSelector(),
                                     pStratFactory->createVectorAverager(),
                                     pStratFactory->createDistanceSumCalculator(), distanceFunc);
    }
};

struct FactoryPair
{
    std::shared_ptr<AbstractKmeansAlgorithmFactory> algFactory;
    std::shared_ptr<AbstractStrategyFactory> stratFactory;
};

class KmeansAlgorithmFactoryProducer
{
public:
    KmeansAlgorithmFactoryProducer(){};

    ~KmeansAlgorithmFactoryProducer(){};

    FactoryPair getAlgFactory(Parallelism parallelism)
    {
        switch (parallelism)
        {
            case (Serial):
                return FactoryPair{ std::make_shared<SharedMemoryKmeansAlgorithmFactory>(new SerialStrategyFactory()),
                                    std::make_shared<SerialStrategyFactory>() };
            case (OMP):
                return FactoryPair{ std::make_shared<SharedMemoryKmeansAlgorithmFactory>(new OMPStrategyFactory()),
                                    std::make_shared<OMPStrategyFactory>() };
            case (MPI):
                return FactoryPair{ std::make_shared<MPIKmeansAlgorithmFactory>(new MPIStrategyFactory()),
                                    std::make_shared<MPIStrategyFactory>() };
            case (Hybrid):
                return FactoryPair{ std::make_shared<MPIKmeansAlgorithmFactory>(new HybridStrategyFactory()),
                                    std::make_shared<HybridStrategyFactory>() };
            default:
                throw std::runtime_error("Invalid parallelism specifier provided.");
        }
    }
};

class KmeansFactory
{
protected:
    std::unique_ptr<KmeansAlgorithmFactoryProducer> pAlgFactoryProducer;

public:
    KmeansFactory() : pAlgFactoryProducer(new KmeansAlgorithmFactoryProducer()){};

    ~KmeansFactory(){};

    AbstractKmeans* createKmeans(Initializer initializer, Maximizer maximizer, CoresetCreator coreset,
                                 Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
                                 const size_t& sampleSize)
    {
        auto factoryPair  = pAlgFactoryProducer->getAlgFactory(parallelism);
        auto algFactory   = factoryPair.algFactory;
        auto stratFactory = factoryPair.stratFactory;

        if (coreset)
        {
            return new CoresetKmeans(
              sampleSize, createKmeans(initializer, maximizer, None, parallelism, distanceFunc, sampleSize),
              algFactory->createCoresetCreator(coreset, sampleSize, distanceFunc),
              stratFactory->createCoresetClusteringFinisher(), stratFactory->createKmeansDataCreator(), distanceFunc);
        }
        else
        {
            return new WeightedKmeans(algFactory->createInitializer(initializer),
                                      algFactory->createMaximizer(maximizer), stratFactory->createKmeansDataCreator(),
                                      distanceFunc);
        }
    }
};