#include "KPlusPlus.hpp"
#include "Utils.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include "mpi.h"
#include <omp.h>

typedef boost::mt19937 RNGType;

void KPlusPlus::initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed)
{
    // initialize RNG
    RNGType rng(seed);
    boost::uniform_int<> intRange(0, matrix->numRows);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // initialize first cluster randomly
    initializeFirstCluster(intDistr());

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int i = 1; i < clusters->numRows; i++)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestCluster(distances, distanceFunc);

        // select point to be next cluster center weighted by nearest distance squared
        weightedClusterSelection(distances, floatDistr());
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestCluster(distances, distanceFunc);
}

void KPlusPlus::initializeFirstCluster(int randIdx)
{
    if (clusters->data.size() != 0)
    {
        throw std::runtime_error(
            "Cannot make call to initializeFirstCluster() when a cluster has already been selected.");
    }

    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    updateClustering(randIdx, 0); // 0 is index of the cluster than has just been added
}

void KPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    for (int i = 0; i < matrix->numRows; i++)
    {
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);
    }
}

void KPlusPlus::weightedClusterSelection(std::vector<value_t> *distances, float randFrac)
{
    int randIdx = weightedRandomSelection(distances, randFrac);
    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    updateClustering(randIdx, getCurrentNumClusters() - 1);
}

void OptimizedKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int clusterIdx = getCurrentNumClusters() - 1;

    for (int i = 0; i < matrix->numRows; i++)
    {
        value_t newDist = (*distanceFunc)(&*matrix->at(i), &*clusters->at(clusterIdx), clusters->numCols);
        if (newDist < distances->at(i) || distances->at(i) < 0)
        {
            updateClustering(i, clusterIdx);
            distances->at(i) = std::pow(newDist, 2);
        }
    }
}

void OMPKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
#pragma omp parallel for shared(distances, distanceFunc), schedule(static)
    for (int i = 0; i < matrix->numRows; i++)
    {
        auto closestCluster = findClosestCluster(i, distanceFunc);
        atomicUpdateClustering(this, i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);
    }
}

void OMPOptimizedKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int clusterIdx = getCurrentNumClusters() - 1;

#pragma omp parallel for shared(distances, distanceFunc, clusterIdx), schedule(static)
    for (int i = 0; i < matrix->numRows; i++)
    {
        value_t newDist = (*distanceFunc)(&*matrix->at(i), &*clusters->at(clusterIdx), clusters->numCols);
        if (newDist < distances->at(i) || distances->at(i) < 0)
        {
            atomicUpdateClustering(this, i, clusterIdx);
            distances->at(i) = std::pow(newDist, 2);
        }
    }
}

void MPIKPlusPlus::initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed)
{
    RNGType rng(seed);
    boost::uniform_int<> intRange(0, matrix->numRows);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // std::vector<value_t> localDistances(matrixChunk->numRows, -1);

    if (rank == 0)
    {
        initializeFirstCluster(intDistr());
    }

    clusters->data.resize(1 * clusters->numCols);
    bcastClusterData(this);

    // initialize remaining clusters; start from index 1 since we know we have only 1 cluster so far
    for (int i = 1; i < clusters->numRows; i++)
    {
        // find distance between each datapoint and nearest cluster, then update clustering assignment
        findAndUpdateClosestCluster(distances, distanceFunc);
        MPI_Allgatherv(MPI_IN_PLACE, lengths->at(rank), MPI_INT, clustering->data(),
                       lengths->data(), displacements->data(), MPI_INT, MPI_COMM_WORLD);

        // aggregate distances and local sums of distances
        // value_t sum, localSum = std::accumulate(localDistances.begin(), localDistances.end(), 0);
        // MPI_Gatherv(localDistances.data(), lengths->at(rank), MPI_FLOAT, distances->data(), lengths->data(),
        //             displacements->data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, lengths->at(rank), MPI_INT, distances->data(),
                       lengths->data(), displacements->data(), MPI_INT, MPI_COMM_WORLD);
        // MPI_Reduce(&localSum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        // select point to be next cluster center weighted by nearest distance squared
        if (rank == 0)
        {
            weightedClusterSelection(distances, floatDistr());
        }

        clusters->data.resize((i + 1) * clusters->numCols);
        bcastClusterData(this);
    }

    // find distance between each datapoint and nearest cluster, then update clustering assignment
    findAndUpdateClosestCluster(distances, distanceFunc);

    MPI_Allgatherv(MPI_IN_PLACE, lengths->at(rank), MPI_INT, clustering->data(),
                   lengths->data(), displacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, lengths->at(rank), MPI_INT, distances->data(),
                   lengths->data(), displacements->data(), MPI_INT, MPI_COMM_WORLD);
    // MPI_Gatherv(localDistances.data(), lengths->at(rank), MPI_FLOAT, distances->data(), lengths->data(),
    //             displacements->data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void MPIKPlusPlus::initializeFirstCluster(int randIdx)
{
    if (clusters->data.size() != 0)
    {
        throw std::runtime_error(
            "Cannot make call to initializeFirstCluster() when a cluster has already been selected.");
    }

    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    AbstractMPIKmeansAlgorithm::updateClustering(this, randIdx, 0); // 0 is index of the cluster than has just been added
}

void MPIKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    for (int i = 0; i < matrixChunk->numRows; i++)
    {
        auto closestCluster = findClosestCluster(&*matrixChunk->at(i), distanceFunc);
        AbstractMPIKmeansAlgorithm::updateClustering(this, displacements->at(rank) + i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);
    }
}
void MPIKPlusPlus::weightedClusterSelection(std::vector<value_t> *distances, float randFrac)
{
    int randIdx = weightedRandomSelection(distances, randFrac);
    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    AbstractMPIKmeansAlgorithm::updateClustering(this, randIdx, getCurrentNumClusters() - 1);
}

void MPIKPlusPlus::setUp(BundledAlgorithmData *bundledData)
{
    AbstractMPIKmeansAlgorithm::setUp(this, dynamic_cast<BundledMPIAlgorithmData *>(bundledData));
}