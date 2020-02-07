#include "Factories/KmeansFactory.hpp"

AbstractKmeans* KmeansFactory::createKmeans(Initializer initializer, Maximizer maximizer, CoresetCreator coreset,
                                            Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
                                            const int_fast32_t& sampleSize)
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

    return new WeightedKmeans(algFactory->createInitializer(initializer), algFactory->createMaximizer(maximizer),
                              stratFactory->createKmeansDataCreator(), distanceFunc);
}