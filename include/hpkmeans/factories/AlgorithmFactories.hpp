#pragma once

#include <hpkmeans/algorithms/coreset/CoresetCreator.hpp>
#include <hpkmeans/algorithms/initializers/KPlusPlus.hpp>
#include <hpkmeans/algorithms/initializers/interface.hpp>
#include <hpkmeans/algorithms/kmeans_algorithm.hpp>
#include <hpkmeans/algorithms/maximizers/Lloyd.hpp>
#include <hpkmeans/algorithms/maximizers/interface.hpp>
#include <hpkmeans/factories/StrategyFactories.hpp>
#include <memory>

namespace HPKmeans
{
template <typename precision, typename int_size>
class AbstractKmeansAlgorithmFactory
{
protected:
    std::unique_ptr<AbstractStrategyFactory<precision, int_size>> pStratFactory;

public:
    AbstractKmeansAlgorithmFactory(AbstractStrategyFactory<precision, int_size>* stratFactory) :
        pStratFactory(stratFactory)
    {
    }

    virtual ~AbstractKmeansAlgorithmFactory() = default;

    IKmeansInitializer<precision, int_size>* createInitializer(Initializer initializer);

    IKmeansMaximizer<precision, int_size>* createMaximizer(Maximizer maximizer);

    AbstractCoresetCreator<precision, int_size>* createCoresetCreator(CoresetCreator coresetCreator,
                                                                      const int_size& sampleSize);

protected:
    virtual IKmeansInitializer<precision, int_size>* getKPP() = 0;

    virtual IKmeansInitializer<precision, int_size>* getOptKPP() = 0;

    virtual IKmeansMaximizer<precision, int_size>* getLloyd() = 0;

    virtual IKmeansMaximizer<precision, int_size>* getOptLloyd() = 0;

    virtual AbstractCoresetCreator<precision, int_size>* getLWCoreset(const int_size& sampleSize) = 0;
};

template <typename precision, typename int_size>
class SharedMemoryKmeansAlgorithmFactory : public AbstractKmeansAlgorithmFactory<precision, int_size>
{
public:
    SharedMemoryKmeansAlgorithmFactory(AbstractStrategyFactory<precision, int_size>* stratFactory) :
        AbstractKmeansAlgorithmFactory<precision, int_size>(stratFactory)
    {
    }

    ~SharedMemoryKmeansAlgorithmFactory() = default;

    IKmeansInitializer<precision, int_size>* getKPP();

    IKmeansInitializer<precision, int_size>* getOptKPP();

    IKmeansMaximizer<precision, int_size>* getLloyd();

    IKmeansMaximizer<precision, int_size>* getOptLloyd();

    AbstractCoresetCreator<precision, int_size>* getLWCoreset(const int_size& sampleSize);
};

template <typename precision, typename int_size>
class MPIKmeansAlgorithmFactory : public AbstractKmeansAlgorithmFactory<precision, int_size>
{
public:
    MPIKmeansAlgorithmFactory(AbstractStrategyFactory<precision, int_size>* stratFactory) :
        AbstractKmeansAlgorithmFactory<precision, int_size>(stratFactory){};

    ~MPIKmeansAlgorithmFactory() = default;

    IKmeansInitializer<precision, int_size>* getKPP();

    IKmeansInitializer<precision, int_size>* getOptKPP();

    IKmeansMaximizer<precision, int_size>* getLloyd();

    IKmeansMaximizer<precision, int_size>* getOptLloyd();

    AbstractCoresetCreator<precision, int_size>* getLWCoreset(const int32_t& sampleSize);
};

template <typename precision, typename int_size>
struct FactoryPair
{
    std::shared_ptr<AbstractKmeansAlgorithmFactory<precision, int_size>> algFactory;
    std::shared_ptr<AbstractStrategyFactory<precision, int_size>> stratFactory;
};

template <typename precision, typename int_size>
class KmeansAlgorithmFactoryProducer
{
public:
    ~KmeansAlgorithmFactoryProducer() = default;

    FactoryPair<precision, int_size> getAlgFactory(Parallelism parallelism);
};

template <typename precision, typename int_size>
IKmeansInitializer<precision, int_size>* AbstractKmeansAlgorithmFactory<precision, int_size>::createInitializer(
  Initializer initializer)
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

template <typename precision, typename int_size>
IKmeansMaximizer<precision, int_size>* AbstractKmeansAlgorithmFactory<precision, int_size>::createMaximizer(
  Maximizer maximizer)
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

template <typename precision, typename int_size>
AbstractCoresetCreator<precision, int_size>* AbstractKmeansAlgorithmFactory<precision, int_size>::createCoresetCreator(
  CoresetCreator coresetCreator, const int_size& sampleSize)
{
    switch (coresetCreator)
    {
        case (LWCoreset):
            return getLWCoreset(sampleSize);
        default:
            return nullptr;
    }
}

template <typename precision, typename int_size>
IKmeansInitializer<precision, int_size>* SharedMemoryKmeansAlgorithmFactory<precision, int_size>::getKPP()
{
    return new SharedMemoryKPlusPlus<precision, int_size>(this->pStratFactory->createClosestClusterUpdater(Reg),
                                                          this->pStratFactory->createWeightedRandomSelector());
}

template <typename precision, typename int_size>
IKmeansInitializer<precision, int_size>* SharedMemoryKmeansAlgorithmFactory<precision, int_size>::getOptKPP()
{
    return new SharedMemoryKPlusPlus<precision, int_size>(this->pStratFactory->createClosestClusterUpdater(Opt),
                                                          this->pStratFactory->createWeightedRandomSelector());
}

template <typename precision, typename int_size>
IKmeansMaximizer<precision, int_size>* SharedMemoryKmeansAlgorithmFactory<precision, int_size>::getLloyd()
{
    return new SharedMemoryLloyd<precision, int_size>(this->pStratFactory->createPointReassigner(Reg),
                                                      this->pStratFactory->createWeightedAverager());
}

template <typename precision, typename int_size>
IKmeansMaximizer<precision, int_size>* SharedMemoryKmeansAlgorithmFactory<precision, int_size>::getOptLloyd()
{
    return new SharedMemoryLloyd<precision, int_size>(this->pStratFactory->createPointReassigner(Opt),
                                                      this->pStratFactory->createWeightedAverager());
}

template <typename precision, typename int_size>
AbstractCoresetCreator<precision, int_size>* SharedMemoryKmeansAlgorithmFactory<precision, int_size>::getLWCoreset(
  const int_size& sampleSize)
{
    return new SharedMemoryCoresetCreator<precision, int_size>(
      sampleSize, this->pStratFactory->createMultiWeightedRandomSelector(), this->pStratFactory->createVectorAverager(),
      this->pStratFactory->createDistanceSumCalculator(), this->pStratFactory->createCoresetDistributionCalculator());
}

template <typename precision, typename int_size>
IKmeansInitializer<precision, int_size>* MPIKmeansAlgorithmFactory<precision, int_size>::getKPP()
{
    return new MPIKPlusPlus<precision, int_size>(this->pStratFactory->createClosestClusterUpdater(Reg),
                                                 this->pStratFactory->createWeightedRandomSelector());
}

template <typename precision, typename int_size>
IKmeansInitializer<precision, int_size>* MPIKmeansAlgorithmFactory<precision, int_size>::getOptKPP()
{
    return new MPIKPlusPlus<precision, int_size>(this->pStratFactory->createClosestClusterUpdater(Opt),
                                                 this->pStratFactory->createWeightedRandomSelector());
}

template <typename precision, typename int_size>
IKmeansMaximizer<precision, int_size>* MPIKmeansAlgorithmFactory<precision, int_size>::getLloyd()
{
    return new MPILloyd<precision, int_size>(this->pStratFactory->createPointReassigner(Reg),
                                             this->pStratFactory->createWeightedAverager());
}

template <typename precision, typename int_size>
IKmeansMaximizer<precision, int_size>* MPIKmeansAlgorithmFactory<precision, int_size>::getOptLloyd()
{
    return new MPILloyd<precision, int_size>(this->pStratFactory->createPointReassigner(Opt),
                                             this->pStratFactory->createWeightedAverager());
}

template <typename precision, typename int_size>
AbstractCoresetCreator<precision, int_size>* MPIKmeansAlgorithmFactory<precision, int_size>::getLWCoreset(
  const int32_t& sampleSize)
{
    return new MPICoresetCreator<precision, int_size>(
      sampleSize, this->pStratFactory->createMultiWeightedRandomSelector(), this->pStratFactory->createVectorAverager(),
      this->pStratFactory->createDistanceSumCalculator());
}

template <typename precision, typename int_size>
FactoryPair<precision, int_size> KmeansAlgorithmFactoryProducer<precision, int_size>::getAlgFactory(
  Parallelism parallelism)
{
    switch (parallelism)
    {
        case (Serial):
            return FactoryPair<precision, int_size>{
                std::make_shared<SharedMemoryKmeansAlgorithmFactory<precision, int_size>>(
                  new SerialStrategyFactory<precision, int_size>()),
                std::make_shared<SerialStrategyFactory<precision, int_size>>()
            };
        case (OMP):
            return FactoryPair<precision, int_size>{
                std::make_shared<SharedMemoryKmeansAlgorithmFactory<precision, int_size>>(
                  new OMPStrategyFactory<precision, int_size>()),
                std::make_shared<OMPStrategyFactory<precision, int_size>>()
            };
        case (MPI):
            return FactoryPair<precision, int_size>{ std::make_shared<MPIKmeansAlgorithmFactory<precision, int_size>>(
                                                       new MPIStrategyFactory<precision, int_size>()),
                                                     std::make_shared<MPIStrategyFactory<precision, int_size>>() };
        case (Hybrid):
            return FactoryPair<precision, int_size>{ std::make_shared<MPIKmeansAlgorithmFactory<precision, int_size>>(
                                                       new HybridStrategyFactory<precision, int_size>()),
                                                     std::make_shared<HybridStrategyFactory<precision, int_size>>() };
        default:
            throw std::runtime_error("Invalid parallelism specifier provided.");
    }
}
}  // namespace HPKmeans