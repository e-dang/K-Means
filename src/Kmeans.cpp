#include "Kmeans.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <time.h>

typedef boost::mt19937 RNGType;

Kmeans::Kmeans(int numClusters, int numRestarts) : numClusters(numClusters), numRestarts(numRestarts)
{
    bestError = INT_MAX;
}

Kmeans::~Kmeans()
{
}

void Kmeans::fit(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &))
{

    int changed;
    int numData = data.size();
    int numFeatures = data[0].size();
    value_t currError;

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
                clusters[i] = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc sum of each feature for all points belonging to a cluster
            for (int i = 0; i < numData; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[clustering[i]].coords[j] += data[i][j];
                }
                clusters[clustering[i]].count++;
            }

            // divide sum by number of points belonging to the cluster to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[i].coords[j] /= clusters[i].count;
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for shared(data), schedule(static), num_threads(8), reduction(+ \
                                                                                   : changed)
            for (int i = 0; i < numData; i++)
            {
                int before = clustering[i];
                nearest(data[i], i, func);
                if (before != clustering[i])
                {
                    changed++;
                }
            }
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change

        // get total sum of distances from each point to their cluster center
        currError = 0;
        for (int i = 0; i < numData; i++)
        {
            currError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
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

void Kmeans::kPlusPlus(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &))
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    value_t sum;
    std::vector<value_t> distances(data.size());

    // initialize first cluster randomly
    clusters.push_back({0, datapoint_t(data[intDistr()])});

    //initialize remaining clusters
    for (int clustIdx = 1; clustIdx < numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        sum = 0;
#pragma omp parallel for shared(data, distances), schedule(static), num_threads(8), reduction(+ \
                                                                                              : sum)
        for (int pointIdx = 0; pointIdx < data.size(); pointIdx++)
        {
            distances[pointIdx] = nearest(data[pointIdx], pointIdx, func);
            sum += distances[pointIdx];
        }

        // select point to be next cluster center weighted by nearest distance squared
        sum *= floatDistr();
        for (int j = 0; j < data.size(); j++)
        {
            if ((sum -= distances[j]) <= 0)
            {
                clusters.push_back({0, datapoint_t(data[j])});
                break;
            }
        }
    }

// assign data points to nearest clusters
#pragma omp parallel for shared(data), schedule(static), num_threads(8)
    for (int i = 0; i < data.size(); i++)
    {
        nearest(data[i], i, func);
    }
}

value_t Kmeans::nearest(datapoint_t &point, int &pointIdx, value_t (*func)(datapoint_t &, datapoint_t &))
{
    value_t tempDist, minDist = INT_MAX - 1;

    // find distance between point and all clusters
    for (int i = 0; i < clusters.size(); i++)
    {
        tempDist = std::pow(func(point, clusters[i].coords), 2);
        if (minDist > tempDist)
        {
            minDist = tempDist;
            clustering[pointIdx] = i;
        }
    }

    return minDist;
}

void Kmeans::fit(dataset_t &data, int overSampling, value_t (*func)(datapoint_t &, datapoint_t &), int initIters)
{

    int changed;
    value_t currError;
    int numFeatures = data[0].size();

    for (int run = 0; run < numRestarts; run++)
    {
        clusters = clusters_t();
        clustering = clustering_t(data.size(), -1);

        // initialize clusters with scalableKmeans algorithm
        std::vector<value_t> closestDists = scaleableKmeans(data, overSampling, func, initIters);

        do
        {
            // reinitialize clusters
            for (int i = 0; i < numClusters; i++)
            {
                clusters[i] = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc sum of each feature for all points belonging to a cluster
            for (int i = 0; i < data.size(); i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[clustering[i]].coords[j] += data[i][j];
                }
                clusters[clustering[i]].count++;
            }

            // divide sum by number of points belonging to the cluster to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[i].coords[j] /= clusters[i].count;
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for shared(data, closestDists), schedule(static), num_threads(8), reduction(+ \
                                                                                                 : changed)
            for (int i = 0; i < data.size(); i++)
            {
                value_t dist = std::pow(func(data[i], clusters[clustering[i]].coords), 2);
                if (dist > closestDists[i] || closestDists[i] < 0)
                {

                    int before = clustering[i];
                    closestDists[i] = nearest(data[i], i, func);
                    if (before != clustering[i])
                    {
                        changed++;
                    }
                }
            }
        } while (changed > (data.size() >> 10)); // do until 99.9% of data doesnt change

        // get total sum of distances from each point to their cluster center
        currError = 0;
        for (int i = 0; i < data.size(); i++)
        {
            currError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
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

std::vector<value_t> Kmeans::scaleableKmeans(dataset_t &data, int &overSampling, value_t (*func)(datapoint_t &, datapoint_t &), int initIters)
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // initialize the closest distances array to large vals
    std::vector<value_t> closestDists(data.size(), INT_MAX);

    // initialize first cluster randomly
    int randIdx = intDistr();
    clusters.push_back({0, datapoint_t(std::begin(data[randIdx]), std::end(data[randIdx]))});

    // select candidate clusters
    value_t sum;
    int prevNumClusters = 0;
    for (int i = 0; i < initIters; i++)
    {
        sum = 0;
#pragma omp parallel for shared(data, closestDists), num_threads(8), schedule(static), reduction(+ \
                                                                                                 : sum)
        for (int i = 0; i < data.size(); i++)
        {
            smartClusterUpdate(data[i], i, prevNumClusters, closestDists, func);
            sum += closestDists[i];
        }
        prevNumClusters = clusters.size();

        // sample each datapoint individually to get an expectation of overSampling new clusters
        for (int j = 0; j < data.size(); j++)
        {
            if (floatDistr() < ((value_t)overSampling) * closestDists[j] / sum)
            {
                clusters.push_back({0, datapoint_t(std::begin(data[j]), std::end(data[j]))});
                clustering[j] = clusters.size() - 1;
            }
        }
    }

    // reassign points to last round of new clusters
#pragma omp parallel for shared(data), num_threads(8), schedule(static)
    for (int i = 0; i < data.size(); i++)
    {
        smartClusterUpdate(data[i], i, prevNumClusters, closestDists, func);
    }

    // weight candidates based on how many points are in each cluster
    std::vector<int> weights(clusters.size(), 0);
    for (int i = 0; i < data.size(); i++)
    {
        weights[clustering[i]]++;
    }

    // get normalizing sum
    sum = 0;
    for (int i = 0; i < weights.size(); i++)
    {
        sum += weights[i];
    }

    // select numClusters clusters from candidates based on weights
    value_t randNum;
    clusters_t selectedClusters = clusters_t();
    clustering_t selectedClusterings = clustering_t(data.size(), -1);
    for (int i = 0; i < numClusters; i++)
    {
        randNum = floatDistr() * sum;
        for (int j = 0; j < clusters.size(); j++)
        {
            if ((randNum -= weights[j]) <= 0)
            {
                sum -= weights[j];
                selectedClusters.push_back(clusters[j]);
                clusters.erase(clusters.begin() + j);
                weights.erase(weights.begin() + j);

#pragma omp parallel for shared(data, selectedClusterings), num_threads(8), schedule(static)
                for (int k = 0; k < data.size(); k++)
                {
                    if (clustering[k] == j)
                    {
                        selectedClusterings[k] = selectedClusters.size() - 1;
                    }
                }
                break;
            }
        }
    }

    // assign data points to nearest clusters
    std::copy(std::begin(selectedClusters), std::end(selectedClusters), clusters);
    std::copy(std::begin(selectedClusterings), std::end(selectedClusterings), clustering);
#pragma omp parallel for shared(data, closestDists), num_threads(8), schedule(static)
    for (int i = 0; i < data.size(); i++)
    {
        if (clustering[i] == -1)
        {
            closestDists[i] = nearest(data[i], i, func);
        }
    }

    return closestDists;
}

void Kmeans::smartClusterUpdate(datapoint_t &point, int &pointIdx, int &clusterIdx, std::vector<value_t> &distances,
                                value_t (*func)(datapoint_t &, datapoint_t &))
{
    value_t tempDist, minDist = INT_MAX - 1;
    int minDistIdx = -1;

    // find the closest new cluster to the point
    for (int i = clusterIdx; i < clusters.size(); i++)
    {
        tempDist = std::pow(func(point, clusters[i].coords), 2);
        if (tempDist < minDist)
        {
            minDist = tempDist;
            minDistIdx = i;
        }
    }

    // compare the closest new cluster to the previously closest cluster and reassign if necessary
    if (minDist < distances[pointIdx])
    {
        distances[pointIdx] = minDist;
        clustering[pointIdx] = minDistIdx;
    }
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

value_t Kmeans::distanceL2(datapoint_t &p1, datapoint_t &p2)
{
    value_t sum = 0;
    for (int i = 0; i < p1.size(); i++)
    {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    return std::sqrt(sum);
}