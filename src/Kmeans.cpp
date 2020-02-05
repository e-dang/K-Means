#include "Kmeans.hpp"

#include "mpi.h"

ClusterResults AbstractKmeans::run(Matrix* data, const int& numClusters, const int& numRestarts, KmeansData* kmeansData)
{
    ClusterResults clusterResults;

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

        compareResults(&clusterData, &distances, &clusterResults);
    }

    return clusterResults;
}

ClusterResults AbstractKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts)
{
    std::vector<value_t> weights(data->getMaxNumData(), 1);
    return fit(data, numClusters, numRestarts, &weights);
}

ClusterResults AbstractKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                   std::vector<value_t>* weights)
{
    auto kmeansData = initKmeansData(data, weights);
    return run(data, numClusters, numRestarts, &kmeansData);
}

KmeansData MPIKmeans::initKmeansData(Matrix* data, std::vector<value_t>* weights)
{
    auto mpiData = getMPIData(mTotalNumData);
    return KmeansData(data, weights, pDistanceFunc, mpiData.rank, mTotalNumData, mpiData.lengths,
                      mpiData.displacements);
}

ClusterResults CoresetKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts)
{
    std::vector<value_t> coresetWeights;
    coresetWeights.reserve(mSampleSize);
    Coreset coreset{ Matrix(mSampleSize, data->getNumFeatures()), coresetWeights };

    pCreator->createCoreset(data, &coreset);

    auto clusterResults = pKmeans->fit(&coreset.data, numClusters, numRestarts, &coreset.weights);
    clusterResults.mClusterData.mClustering.resize(mTotalNumData);
    clusterResults.mSqDistances.resize(mTotalNumData);

    pCreator->finishClustering(data, &clusterResults);

    return clusterResults;
}

ClusterResults CoresetKmeans::fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                  std::vector<value_t>* weights)
{
    throw std::runtime_error("Should not be calling this func.");
    return ClusterResults{};
}