#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "Kmeans.hpp"
#include "KmeansFactories.hpp"

class Kmeans
{
protected:
    std::unique_ptr<AbstractKmeans> pKmeans;
    std::unique_ptr<KmeansFactory> pFactory;

    Initializer mInitializer;
    Maximizer mMaximizer;
    CoresetCreator mCoreset;
    Parallelism mParallelism;
    int_fast32_t mSampleSize;

public:
    Kmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
           const Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
           const int_fast32_t sampleSize = -1) :
        pKmeans(nullptr),
        pFactory(new KmeansFactory()),
        mInitializer(initializer),
        mMaximizer(maximizer),
        mCoreset(coreset),
        mParallelism(parallelism),
        mSampleSize(sampleSize)
    {
        setKmeans(initializer, maximizer, coreset, parallelism, distanceFunc, sampleSize);
    }

    ~Kmeans(){};

    std::shared_ptr<ClusterResults> fit(const Matrix* const data, const int_fast32_t& numClusters,
                                        const int& numRestarts)
    {
        if (isValidSampleSize(data) && pKmeans != nullptr)
            return pKmeans->fit(data, numClusters, numRestarts);

        return nullptr;
    }

    std::shared_ptr<ClusterResults> fit(const Matrix* const data, const int_fast32_t& numClusters,
                                        const int& numRestarts, std::vector<value_t>* weights)
    {
        if (isValidSampleSize(data) && pKmeans != nullptr)
            return pKmeans->fit(data, numClusters, numRestarts, weights);

        return nullptr;
    }

    void setDistanceFunc(IDistanceFunctor* distanceFunc) { pKmeans->setDistanceFunc(distanceFunc); }

    bool setKmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
                   const Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
                   const int_fast32_t sampleSize = -1)
    {
        if (sampleSizeCheck())
        {
            setInitializer(initializer);
            setMaximizer(maximizer);
            setCoresetCreator(coreset);
            setParallelism(parallelism);
            pKmeans = std::unique_ptr<AbstractKmeans>(
              pFactory->createKmeans(initializer, maximizer, coreset, parallelism, distanceFunc, sampleSize));

            return true;
        }

        return false;
    }

private:
    void setInitializer(const Initializer initializer) { mInitializer = initializer; }
    void setMaximizer(const Maximizer maximizer) { mMaximizer = maximizer; }
    void setCoresetCreator(const CoresetCreator coreset) { mCoreset = coreset; }
    void setParallelism(const Parallelism parallelism) { mParallelism = parallelism; }

    bool sampleSizeCheck()
    {
        if (mCoreset == None || (mCoreset != None && mSampleSize >= 0))
            return true;

        std::cout << "A sample size must be given when using CoresetKmeans." << std::endl;
        return false;
    }

    bool isValidSampleSize(const Matrix* const data)
    {
        if (mSampleSize <= data->getNumData())
            return true;

        return false;
    }
};