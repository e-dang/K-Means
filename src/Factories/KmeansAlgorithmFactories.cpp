#include "Factories/KmeansAlgorithmFactories.hpp"

AbstractKmeansInitializer* AbstractKmeansAlgorithmFactory::createInitializer(Initializer initializer)
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

AbstractKmeansMaximizer* AbstractKmeansAlgorithmFactory::createMaximizer(Maximizer maximizer)
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

AbstractCoresetCreator* AbstractKmeansAlgorithmFactory::createCoresetCreator(
  CoresetCreator coresetCreator, const int32_t& sampleSize, std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    switch (coresetCreator)
    {
        case (LWCoreset):
            return getLWCoreset(sampleSize, distanceFunc);
        default:
            return nullptr;
    }
}

FactoryPair KmeansAlgorithmFactoryProducer::getAlgFactory(Parallelism parallelism)
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