#include "Kmeans.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <time.h>

Kmeans::Kmeans(int numClusters, int numRestarts) : numClusters(numClusters), numRestarts(numRestarts)
{
    bestError = INT_MAX;
}

Kmeans::~Kmeans()
{
}

void Kmeans::kPlusPlus(dataset_t data, float (*func)(datapoint_t &, datapoint_t &))
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    float sum;
    std::vector<float> distances(data.size());

    // initialize first cluster randomly
    clusters.push_back({0, datapoint_t(data.at(intDistr()))});

    //initialize remaining clusters
    for (int clustIdx = 1; clustIdx < numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        sum = 0;
#pragma omp parallel for shared(data, distances), schedule(static), num_threads(8), reduction(+ \
                                                                                              : sum)
        for (int pointIdx = 0; pointIdx < data.size(); pointIdx++)
        {
            distances[pointIdx] = std::pow(nearest(data.at(pointIdx), pointIdx, func), 2);
            sum += distances[pointIdx];
        }

        // select point to be next cluster center weighted by nearest distance squared
        sum *= floatDistr();
        for (int j = 0; j < data.size(); j++)
        {
            if ((sum -= distances[j]) <= 0)
            {
                clusters.push_back({0, datapoint_t(data.at(j))});
                break;
            }
        }
    }

// assign data points to nearest clusters
#pragma omp parallel for shared(data), schedule(static), num_threads(8)
    for (int i = 0; i < data.size(); i++)
    {
        nearest(data.at(i), i, func);
    }
}

float Kmeans::nearest(datapoint_t &point, int &pointIdx, float (*func)(datapoint_t &, datapoint_t &))
{
    float tempDist, minDist = INT_MAX - 1;

    // find distance between point and all clusters
    for (int i = 0; i < clusters.size(); i++)
    {
        tempDist = func(point, clusters.at(i).coords);

        if (minDist > tempDist)
        {
            minDist = tempDist;
            clustering.at(pointIdx) = i;
        }
    }

    return minDist;
}

bool Kmeans::setNumClusters(int numClusters)
{
    this->numClusters = numClusters;
    return true;
}

bool Kmeans::setNumRestarts(int numRestarts)
{
    this->numRestarts = numRestarts;
    return true;
}