#include "Kmeans.hpp"

#include "mpi.h"

std::shared_ptr<ClusterResults> AbstractKmeans::run(Matrix* data, const int& numClusters, const int& numRestarts,
                                                    KmeansData* kmeansData)
{
    std::shared_ptr<ClusterResults> clusterResults = std::make_shared<ClusterResults>();

    pInitializer->setKmeansData(kmeansData);
    pMaximizer->setKmeansData(kmeansData);

    for (int i = 0; i < numRestarts; i++)
    {
        std::vector<value_t> distances(kmeansData->mTotalNumData, 1);
        ClusterData clusterData(kmeansData->mTotalNumData, data->getNumFeatures(), numClusters);

        kmeansData->setClusterData(&clusterData);
        kmeansData->setSqDistances(&distances);

        pInitializer->initialize();
        pMaximizer->maximize();

        compareResults(&clusterData, &distances, clusterResults);
    }

    return clusterResults;
}

std::shared_ptr<ClusterResults> WeightedKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts)
{
    std::vector<value_t> weights(data->getMaxNumData(), 1);
    return fit(data, numClusters, numRestarts, &weights);
}

std::shared_ptr<ClusterResults> WeightedKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                                    std::vector<value_t>* weights)
{
    auto kmeansData = pDataCreator->create(data, weights, pDistanceFunc);
    return run(data, numClusters, numRestarts, &kmeansData);
}

std::shared_ptr<ClusterResults> CoresetKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts)
{
    auto kmeansData = pDataCreator->create(data, nullptr, pDistanceFunc);

    std::vector<value_t> coresetWeights;
    coresetWeights.reserve(mSampleSize);
    Coreset coreset{ Matrix(mSampleSize, data->getNumFeatures()), coresetWeights };

    pCreator->createCoreset(data, &coreset);

    auto clusterResults = pKmeans->fit(&coreset.data, numClusters, numRestarts, &coreset.weights);
    clusterResults->mClusterData.mClustering.resize(kmeansData.mTotalNumData);
    clusterResults->mSqDistances.resize(kmeansData.mTotalNumData);
    std::fill(clusterResults->mClusterData.mClustering.begin(), clusterResults->mClusterData.mClustering.end(), -1);
    std::fill(clusterResults->mSqDistances.begin(), clusterResults->mSqDistances.end(), -1);

    kmeansData.setClusterData(&clusterResults->mClusterData);
    kmeansData.setSqDistances(&clusterResults->mSqDistances);

    clusterResults->mError = pFinisher->finishClustering(&kmeansData);

    return clusterResults;
}

std::shared_ptr<ClusterResults> CoresetKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                                   std::vector<value_t>* weights)
{
    throw std::runtime_error("Should not be calling this func.");
    return nullptr;
}