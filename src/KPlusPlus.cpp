#include "KPlusPlus.hpp"
#include "Utils.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

typedef boost::mt19937 RNGType;

void SerialKPlusPlus::initialize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc, const float &seed)
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

void SerialKPlusPlus::initializeFirstCluster(int randIdx)
{
    if (clusters->data.size() != 0)
    {
        throw std::runtime_error(
            "Cannot make call to initializeFirstCluster() when a cluster has already been selected.");
    }

    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    updateClustering(randIdx, 0); // 0 is index of the cluster than has just been added
}

void SerialKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    for (int i = 0; i < matrix->numRows; i++)
    {
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);
    }
}

void SerialKPlusPlus::weightedClusterSelection(std::vector<value_t> *distances, float randFrac)
{
    int randIdx = weightedRandomSelection(distances, randFrac);
    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    updateClustering(randIdx, getCurrentNumClusters() - 1);
}

void OptimizedSerialKPlusPlus::findAndUpdateClosestCluster(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    for (int i = 0; i < matrix->numRows; i++)
    {
        auto closestCluster = findClosestCluster(i, distanceFunc);
        if (closestCluster.distance < distances->at(i) || distances->at(i) < 0)
        {
            updateClustering(i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);
        }
    }
}

ClosestCluster OptimizedSerialKPlusPlus::findClosestCluster(const int &dataIdx, IDistanceFunctor *distanceFunc)
{
    int clusterIdx = getCurrentNumClusters() - 1;
    value_t distance = (*distanceFunc)(&*matrix->at(dataIdx), &*clusters->at(clusterIdx), clusters->numCols);

    return ClosestCluster{clusterIdx, distance};
}