#include "KPlusPlus.hpp"
#include "Utils.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

typedef boost::mt19937 RNGType;

void SerialKPlusPlus::initialize(IDistanceFunctor *distanceFunc, const float &seed)
{
    // initialize RNG
    RNGType rng(seed);
    boost::uniform_int<> intRange(0, matrix->numRows);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // initialize first cluster randomly
    int randIdx = intDistr();
    int numExistingClusters = 0;
    std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
    updateClustering(randIdx, numExistingClusters);
    numExistingClusters++;

    //initialize remaining clusters
    std::vector<value_t> distances(matrix->numRows, -1);
    for (int i = numExistingClusters; i < clusters->numRows; i++)
    {
        // find distance between each data point and nearest cluster, then update clustering assignments
        for (int j = 0; j < matrix->numRows; j++)
        {
            auto closestCluster = findClosestCluster(&*matrix->at(j), clusters, distanceFunc);
            updateClustering(j, closestCluster.clusterIdx);
            distances[j] = std::pow(closestCluster.distance, 2);
        }

        // select point to be next cluster center weighted by nearest distance squared
        randIdx = weightedRandomSelection(&distances, floatDistr());
        std::copy(matrix->at(randIdx), matrix->at(randIdx) + matrix->numCols, std::back_inserter(clusters->data));
        updateClustering(randIdx, numExistingClusters);
        numExistingClusters++;
    }

    // assign data points to nearest clusters
    for (int i = 0; i < matrix->numRows; i++)
    {
        auto closestPoint = findClosestCluster(&*matrix->at(i), clusters, distanceFunc);
        updateClustering(i, closestPoint.clusterIdx);
    }
}