#include "Kmeans/KmeansFacade.hpp"

Kmeans::Kmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
               const Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
               const int32_t sampleSize) :
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

std::shared_ptr<ClusterResults> Kmeans::fit(const Matrix* const data, const int32_t& numClusters,
                                            const int& numRestarts)
{
    if (isValidSampleSize(data) && pKmeans != nullptr)
        return pKmeans->fit(data, numClusters, numRestarts);

    return nullptr;
}

std::shared_ptr<ClusterResults> Kmeans::fit(const Matrix* const data, const int32_t& numClusters,
                                            const int& numRestarts, std::vector<value_t>* weights)
{
    if (isValidSampleSize(data) && pKmeans != nullptr)
        return pKmeans->fit(data, numClusters, numRestarts, weights);

    return nullptr;
}

bool Kmeans::setKmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
                       const Parallelism parallelism, std::shared_ptr<IDistanceFunctor> distanceFunc,
                       const int32_t sampleSize)
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

bool Kmeans::sampleSizeCheck()
{
    if (mCoreset == None || (mCoreset != None && mSampleSize >= 0))
        return true;

    std::cout << "A sample size must be given when using CoresetKmeans." << std::endl;
    return false;
}

bool Kmeans::isValidSampleSize(const Matrix* const data)
{
    if (mSampleSize <= data->getNumData())
        return true;

    return false;
}