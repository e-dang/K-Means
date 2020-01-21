#include "KPlusPlus.hpp"
#include "Utils.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include "mpi.h"
#include <omp.h>

typedef boost::mt19937 RNGType;

void TemplateKPlusPlus::initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed)
{
    // initialize RNG
    RNGType rng(seed);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // initialize first cluster uniformly at random. Thus distances should be filled with same number i.e. 1
    weightedClusterSelection(distances, floatDistr());

    // change fill distances vector with -1 so values aren't confused with actual distances
    std::fill(distances->begin(), distances->end(), -1);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    int numClusters = pClusters->numRows;
    for (int i = 1; i < numClusters; i++)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestCluster(distances, distanceFunc);

        // select point to be next cluster center weighted by nearest distance squared
        weightedClusterSelection(distances, floatDistr());
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestCluster(distances, distanceFunc);
}

void TemplateKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int numData = pMatrix->numRows;
    for (int i = 0; i < numData; i++)
    {
        pFinder->findAndUpdateClosestCluster(i, distances, distanceFunc);
    }
}

void TemplateKPlusPlus::weightedClusterSelection(std::vector<value_t> *distances, float randFrac)
{
    float randSumFrac = randFrac * std::accumulate(distances->begin(), distances->end(), 0);
    pSelector->weightedClusterSelection(distances, randSumFrac);
}

void OMPKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int numData = pMatrix->numRows;
#pragma omp parallel for shared(numData, distances, distanceFunc), schedule(static)
    for (int i = 0; i < numData; i++)
    {
        pFinder->findAndUpdateClosestCluster(i, distances, distanceFunc);
    }
}

void MPIKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int numData = pMatrixChunk->numRows;
    for (int i = 0; i < numData; i++)
    {
        pFinder->findAndUpdateClosestCluster(pDisplacements->at(mRank) + i, distances, distanceFunc);
    }

    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
                   pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_FLOAT, distances->data(),
                   pLengths->data(), pDisplacements->data(), MPI_FLOAT, MPI_COMM_WORLD);
}

void MPIKPlusPlus::weightedClusterSelection(std::vector<value_t> *distances, float randFrac)
{
    // for (auto iter = distances->begin() + pDisplacements->at(mRank); iter != distances->begin() + pDisplacements->at(mRank) + pLengths->at(mRank); iter++)
    // {
    //     std::cout << *iter << " ";
    // }
    // std::cout << std::endl
    //           << std::endl;
    // auto start = distances->begin() + pDisplacements->at(mRank);
    // auto stop = start + pLengths->at(mRank);
    // float sum, localSum = std::accumulate(start, stop, 0);
    // std::vector<value_t> sums()
    // MPI_Reduce(&localSum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mRank == 0)
    {
        value_t sum = std::accumulate(distances->begin(), distances->end(), 0);
        float randSumFrac = sum * randFrac;
        pSelector->weightedClusterSelection(distances, randSumFrac);
    }
    else
    {
        pClusters->data.resize((getCurrentNumClusters() + 1) * pClusters->numCols);
    }

    bcastClusterData();
}

// void MPIKPlusPlus::initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed)
// {
//     RNGType rng(seed);
//     boost::uniform_int<> intRange(0, pMatrix->numRows);
//     boost::uniform_real<> floatRange(0, 1);
//     boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
//     boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

//     // std::vector<value_t> localDistances(matrixChunk->numRows, -1);

//     if (mRank == 0)
//     {
//         initializeFirstCluster(intDistr());
//     }

//     pClusters->data.resize(1 * pClusters->numCols);
//     bcastClusterData();

//     // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
//     for (int i = 1; i < pClusters->numRows; i++)
//     {
//         // find distance between each datapoint and nearest cluster, then update clustering assignment
//         findAndUpdateClosestCluster(distances, distanceFunc);
//         MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
//                        pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);

//         // aggregate distances and local sums of distances
//         // value_t sum, localSum = std::accumulate(localDistances.begin(), localDistances.end(), 0);
//         // MPI_Gatherv(localDistances.data(), pLengths->at(mRank), MPI_FLOAT, distances->data(), pLengths->data(),
//         //             pDisplacements->data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, distances->data(),
//                        pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
//         // MPI_Reduce(&localSum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

//         // select point to be next cluster center weighted by nearest distance squared
//         if (mRank == 0)
//         {
//             weightedClusterSelection(distances, floatDistr());
//         }

//         pClusters->data.resize((i + 1) * pClusters->numCols);
//         bcastClusterData();
//     }

//     // find distance between each datapoint and nearest cluster, then update clustering assignment
//     findAndUpdateClosestCluster(distances, distanceFunc);

//     MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, pClustering->data(),
//                    pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
//     MPI_Allgatherv(MPI_IN_PLACE, pLengths->at(mRank), MPI_INT, distances->data(),
//                    pLengths->data(), pDisplacements->data(), MPI_INT, MPI_COMM_WORLD);
//     // MPI_Gatherv(localDistances.data(), pLengths->at(mRank), MPI_FLOAT, distances->data(), pLengths->data(),
//     //             pDisplacements->data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
// }

// void MPIKPlusPlus::initializeFirstCluster(int randIdx)
// {
//     if (pClusters->data.size() != 0)
//     {
//         throw std::runtime_error(
//             "Cannot make call to initializeFirstCluster() when a cluster has already been selected.");
//     }

//     std::copy(pMatrix->at(randIdx), pMatrix->at(randIdx) + pMatrix->numCols, std::back_inserter(pClusters->data));
//     AbstractMPIKmeansAlgorithm::updateClustering(randIdx, 0); // 0 is index of the cluster than has just been added
// }

// void MPIKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
// {
//     for (int i = 0; i < pMatrixChunk->numRows; i++)
//     {
//         auto closestCluster = findClosestCluster(i, distanceFunc);
//         AbstractMPIKmeansAlgorithm::updateClustering(pDisplacements->at(mRank) + i, closestCluster.clusterIdx);
//         distances->at(i) = std::pow(closestCluster.distance, 2);
//     }
// }
// void MPIKPlusPlus::weightedClusterSelection(std::vector<value_t> *distances, float randFrac)
// {
//     int randIdx = weightedRandomSelection(distances, randFrac);
//     std::copy(pMatrix->at(randIdx), pMatrix->at(randIdx) + pMatrix->numCols, std::back_inserter(pClusters->data));
//     AbstractMPIKmeansAlgorithm::updateClustering(randIdx, getCurrentNumClusters() - 1);
// }