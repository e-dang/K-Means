#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Factories/KmeansFactory.hpp"
#include "Kmeans/Kmeans.hpp"

class Kmeans
{
protected:
    std::unique_ptr<AbstractKmeans> pKmeans;
    std::unique_ptr<KmeansFactory> pFactory;

    Initializer mInitializer;
    Maximizer mMaximizer;
    CoresetCreator mCoreset;
    Parallelism mParallelism;
    int32_t mSampleSize;

public:
    Kmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
           const Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
           const int32_t sampleSize = -1);

    ~Kmeans(){};

    std::shared_ptr<ClusterResults> fit(const Matrix* const data, const int32_t& numClusters, const int& numRestarts);

    std::shared_ptr<ClusterResults> fit(const Matrix* const data, const int32_t& numClusters, const int& numRestarts,
                                        std::vector<value_t>* weights);

    bool setKmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
                   const Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
                   const int32_t sampleSize = -1);

    void setDistanceFunc(IDistanceFunctor* distanceFunc) { pKmeans->setDistanceFunc(distanceFunc); }

private:
    void setInitializer(const Initializer initializer) { mInitializer = initializer; }

    void setMaximizer(const Maximizer maximizer) { mMaximizer = maximizer; }

    void setCoresetCreator(const CoresetCreator coreset) { mCoreset = coreset; }

    void setParallelism(const Parallelism parallelism) { mParallelism = parallelism; }

    bool sampleSizeCheck();

    bool isValidSampleSize(const Matrix* const data);
};