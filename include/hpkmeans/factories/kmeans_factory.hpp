#pragma once

#include <hpkmeans/factories/algorithm_factories.hpp>
#include <hpkmeans/factories/state_factories.hpp>
#include <hpkmeans/kmeans/coreset_kmeans.hpp>
#include <hpkmeans/kmeans/weighted_kmeans.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
class KmeansFactory
{
private:
    KmeansAlgorithmFactoryProducer<precision, int_size> m_AlgFactoryProducer;
    KmeansStateAbstractFactory<precision, int_size> m_StateAbstractFactory;

public:
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
    auto factoryPair  = m_AlgFactoryProducer.getAlgFactory(parallelism);
    auto algFactory   = factoryPair.algFactory;
    auto stratFactory = factoryPair.stratFactory;

    if (coreset)
    {
        return new CoresetKmeansWrapper<precision, int_size>(
          sampleSize, createKmeans(initializer, maximizer, None, parallelism, distanceFunc, sampleSize),
          algFactory->createCoresetCreator(coreset, sampleSize), stratFactory->createCoresetClusteringFinisher(),
          m_StateAbstractFactory.createStateFactory(parallelism), distanceFunc);
    }

    return new WeightedKmeansWrapper<precision, int_size>(
      algFactory->createInitializer(initializer), algFactory->createMaximizer(maximizer),
      m_StateAbstractFactory.createStateFactory(parallelism), distanceFunc);
}
}  // namespace HPKmeans