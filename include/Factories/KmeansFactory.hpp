#pragma once

#include "Factories/KmeansAlgorithmFactories.hpp"
#include "Kmeans/KmeansWrappers.hpp"

namespace HPKmeans
{
template <typename precision, typename int_size>
class KmeansFactory
{
protected:
    std::unique_ptr<KmeansAlgorithmFactoryProducer<precision, int_size>> pAlgFactoryProducer;

public:
    KmeansFactory() : pAlgFactoryProducer(new KmeansAlgorithmFactoryProducer<precision, int_size>()) {}

    ~KmeansFactory() = default;

    AbstractKmeansWrapper<precision, int_size>* createKmeans(Initializer initializer, Maximizer maximizer,
                                                             CoresetCreator coreset, Parallelism parallelism,
                                                             std::shared_ptr<IDistanceFunctor<precision>> distanceFunc,
                                                             const int_size& sampleSize);
};

template <typename precision, typename int_size>
AbstractKmeansWrapper<precision, int_size>* KmeansFactory<precision, int_size>::createKmeans(
  Initializer initializer, Maximizer maximizer, CoresetCreator coreset, Parallelism parallelism,
  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc, const int_size& sampleSize)
{
    auto factoryPair  = pAlgFactoryProducer->getAlgFactory(parallelism);
    auto algFactory   = factoryPair.algFactory;
    auto stratFactory = factoryPair.stratFactory;

    if (coreset)
    {
        return new CoresetKmeansWrapper<precision, int_size>(
          sampleSize, createKmeans(initializer, maximizer, None, parallelism, distanceFunc, sampleSize),
          algFactory->createCoresetCreator(coreset, sampleSize), stratFactory->createCoresetClusteringFinisher(),
          stratFactory->createKmeansStateInitializer(), distanceFunc);
    }

    return new WeightedKmeansWrapper<precision, int_size>(algFactory->createInitializer(initializer),
                                                          algFactory->createMaximizer(maximizer),
                                                          stratFactory->createKmeansStateInitializer(), distanceFunc);
}
}  // namespace HPKmeans