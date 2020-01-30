#include "KPlusPlus.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include "mpi.h"
#include <omp.h>

typedef boost::mt19937 RNGType;

void TemplateKPlusPlus::initialize(const float &seed)
{
    // initialize RNG
    RNGType rng(seed);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // initialize first cluster uniformly at random. Thus distances should be filled with same number i.e. 1
    weightedClusterSelection(floatDistr());

    // change fill distances vector with -1 so values aren't confused with actual distances
    std::fill(pDistances->begin(), pDistances->end(), -1);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int i = 1; i < pClusters->getMaxNumData(); i++)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestCluster();

        // select point to be next cluster center weighted by nearest distance squared
        weightedClusterSelection(floatDistr());
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestCluster();
}

void TemplateKPlusPlus::weightedClusterSelection(float randFrac)
{
    float randSumFrac = randFrac * std::accumulate(pDistances->begin(), pDistances->end(), 0);
    int dataIdx = pSelector->select(pDistances, randSumFrac);
    pClusters->appendDataPoint(pData->at(dataIdx));
}

void TemplateKPlusPlus::findAndUpdateClosestCluster()
{
    for (int i = 0; i < pData->getNumData(); i++)
    {
        auto closestCluster = pFinder->findClosestCluster(pData->at(i), pDistanceFunc);
        pUpdater->update(i, closestCluster.clusterIdx, pWeights->at(i));
        pDistances->at(i) = closestCluster.distance;
    }
}

void OMPKPlusPlus::findAndUpdateClosestCluster()
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < pData->getNumData(); i++)
    {
        auto closestCluster = pFinder->findClosestCluster(pData->at(i), pDistanceFunc);
        pUpdater->update(i, closestCluster.clusterIdx, pWeights->at(i));
        pDistances->at(i) = closestCluster.distance;
    }
}

// void MPIKPlusPlus::findAndUpdateClosestCluster(std::vector<int> *clustering, std::vector<value_t> *clusterWeights,
//                                                std::vector<value_t> *distances)
// {
//     for (int i = 0; i < pData->getNumData(); i++)
//     {
//         pFinder->findAndUpdateClosestCluster(pDisplacements->at(mRank) + i, distances, distanceFunc);
//     }

//     MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
//                    pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
//     MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_FLOAT, distances->data(),
//                    pLengths->data(), pDisplacements->data(), MPI_FLOAT, MPI_COMM_WORLD);
// }

// void MPIKPlusPlus::weightedClusterSelection(Matrix *clusters, std::vector<value_t> *distances, float randFrac)
// {
//     // for (auto iter = distances->begin() + pDisplacements->at(mRank); iter != distances->begin() + pDisplacements->at(mRank) + pLengths->at(mRank); iter++)
//     // {
//     //     std::cout << *iter << " ";
//     // }
//     // std::cout << std::endl
//     //           << std::endl;
//     // auto start = distances->begin() + pDisplacements->at(mRank);
//     // auto stop = start + pLengths->at(mRank);
//     // float sum, localSum = std::accumulate(start, stop, 0);
//     // std::vector<value_t> sums()
//     // MPI_Reduce(&localSum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

//     if (mRank == 0)
//     {
//         value_t sum = std::accumulate(distances->begin(), distances->end(), 0);
//         float randSumFrac = sum * randFrac;
//         pSelector->weightedClusterSelection(distances, randSumFrac);
//     }
//     else
//     {
//         pClusters->data.resize((getCurrentNumClusters() + 1) * pClusters->numCols);
//     }

//     bcastClusterData();
// }