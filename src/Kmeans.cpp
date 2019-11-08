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

void Kmeans::fit(dataset_t data, float (*func)(datapoint_t &, datapoint_t &))
{

    int changed;
    int numData = data.size();
    int numFeatures = data.at(0).size();
    float currError;

    for (int run = 0; run < numRestarts; run++)
    {
        clusters = clusters_t();
        clustering = clustering_t(numData, -1);

        // initialize clusters with k++ algorithm
        kPlusPlus(data, func);

        do
        {
            // reinitialize clusters
            for (int i = 0; i < numClusters; i++)
            {
                clusters.at(i) = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc sum of each feature for all points belonging to a cluster
            for (int i = 0; i < numData; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters.at(clustering.at(i)).coords[j] += data.at(i)[j];
                }
                clusters.at(clustering.at(i)).count++;
            }

            // divide sum by number of points belonging to the cluster to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters.at(i).coords[j] /= clusters.at(i).count;
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for shared(data), schedule(static), num_threads(8), reduction(+ \
                                                                                   : changed)
            for (int i = 0; i < numData; i++)
            {
                int before = clustering.at(i);
                nearest(data.at(i), i, func);
                if (before != clustering.at(i))
                {
                    changed++;
                }
            }
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change

        // get total sum of distances from each point to their cluster center
        currError = 0;
        for (int i = 0; i < numData; i++)
        {
            currError += func(data.at(i), clusters.at(clustering.at(i)).coords);
        }

        // if this round produced lowest error, keep clustering
        if (currError < bestError)
        {
            bestError = currError;
            std::copy(std::begin(clustering), std::end(clustering), bestClustering);
            std::copy(std::begin(clusters), std::end(clusters), bestClusters);
        }
    }
}

void Kmeans::kPlusPlus(dataset_t &data, float (*func)(datapoint_t &, datapoint_t &))
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