#pragma once

#include <memory>

#include "CoresetCreator.hpp"
#include "Factories/StrategyFactories.hpp"
#include "KPlusPlus.hpp"
#include "KmeansAlgorithms.hpp"
#include "Lloyd.hpp"

class AbstractKmeansAlgorithmFactory
{
protected:
    std::unique_ptr<AbstractStrategyFactory> pStratFactory;

public:
    AbstractKmeansAlgorithmFactory(AbstractStrategyFactory* stratFactory) : pStratFactory(stratFactory){};

    virtual ~AbstractKmeansAlgorithmFactory(){};

    AbstractKmeansInitializer* createInitializer(Initializer initializer);

    AbstractKmeansMaximizer* createMaximizer(Maximizer maximizer);

    AbstractCoresetCreator* createCoresetCreator(CoresetCreator coresetCreator, const int_fast32_t& sampleSize,
                                                 std::shared_ptr<IDistanceFunctor> distanceFunc);

protected:
    virtual AbstractKmeansInitializer* getKPP() = 0;

    virtual AbstractKmeansInitializer* getOptKPP() = 0;

    virtual AbstractKmeansMaximizer* getLloyd() = 0;

    virtual AbstractKmeansMaximizer* getOptLloyd() = 0;

    virtual AbstractCoresetCreator* getLWCoreset(const int_fast32_t& sampleSize,
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

    AbstractCoresetCreator* getLWCoreset(const int_fast32_t& sampleSize, std::shared_ptr<IDistanceFunctor> distanceFunc)
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

    AbstractCoresetCreator* getLWCoreset(const int_fast32_t& sampleSize, std::shared_ptr<IDistanceFunctor> distanceFunc)
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

    FactoryPair getAlgFactory(Parallelism parallelism);
};