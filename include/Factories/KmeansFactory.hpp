#pragma once

#include "Factories/KmeansAlgorithmFactories.hpp"
#include "Kmeans.hpp"

class KmeansFactory
{
protected:
    std::unique_ptr<KmeansAlgorithmFactoryProducer> pAlgFactoryProducer;

public:
    KmeansFactory() : pAlgFactoryProducer(new KmeansAlgorithmFactoryProducer()){};

    ~KmeansFactory(){};

    AbstractKmeans* createKmeans(Initializer initializer, Maximizer maximizer, CoresetCreator coreset,
                                 Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
                                 const int_fast32_t& sampleSize);
};