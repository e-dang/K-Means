#include "KPlusPlus.hpp"
#include "Utils.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
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
#pragma omp parallel for shared(distances), schedule(static)
    for (int i = 0; i < matrix->numRows; i++)
    {
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);
    }
}

inline void OMPKPlusPlus::updateClustering(const int &dataIdx, const int &clusterIdx)
{
    int &clusterAssignment = clustering->at(dataIdx);

    // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
    if (clusterAssignment >= 0 && clusterWeights->at(clusterAssignment) > 0)
#pragma omp atomic
        clusterWeights->at(clusterAssignment) -= weights->at(dataIdx);
#pragma omp atomic
    clusterWeights->at(clusterIdx) += weights->at(dataIdx);
    clusterAssignment = clusterIdx;
}

void OMPOptimizedKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int clusterIdx = getCurrentNumClusters() - 1;

#pragma omp parallel for shared(distances, clusterIdx), schedule(static)
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