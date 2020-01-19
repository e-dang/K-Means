
#include "Lloyd.hpp"

void Lloyd::maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed;
    do
    {
        updateClusters();
        changed = reassignPoints(distances, distanceFunc);

    } while (changed > (matrix->numRows * MIN_PERCENT_CHANGED)); // do until 99.9% of data doesnt change
}

void Lloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(clusters->data.begin(), clusters->data.end(), 0);

    // calc the weighted sum of each feature for all points belonging to a cluster
    for (int i = 0; i < matrix->numRows; i++)
    {
        for (int j = 0; j < matrix->numCols; j++)
        {
            clusters->at(clustering->at(i), j) += weights->at(i) * matrix->at(i, j);
        }
    }

    // average out the weighted sum of each cluster based on the number of datapoints assigned to it
    for (int i = 0; i < clusters->numRows; i++)
    {
        for (int j = 0; j < clusters->numCols; j++)
        {
            clusters->data.at(i * clusters->numCols + j) /= clusterWeights->at(i);
        }
    }
}

int Lloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < matrix->numRows; i++)
    {

        // find closest cluster for each datapoint and update cluster assignment
        int before = clustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != clustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < matrix->numRows; i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        value_t dist = std::pow((*distanceFunc)(&*matrix->at(i), &*clusters->at(clustering->at(i)), matrix->numCols), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            // find closest cluster for each datapoint and update cluster assignment
            int before = clustering->at(i);
            auto closestCluster = findClosestCluster(i, distanceFunc);
            updateClustering(i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);

            // check if cluster assignments have changed
            if (before != clustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            distances->at(i) = dist;
        }
    }

    return changed;
}

void OMPLloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(clusters->data.begin(), clusters->data.end(), 0);

    // calc the weighted sum of each feature for all points belonging to a cluster
    for (int i = 0; i < matrix->numRows; i++)
    {
        for (int j = 0; j < matrix->numCols; j++)
        {
            clusters->at(clustering->at(i), j) += weights->at(i) * matrix->at(i, j);
        }
    }

    // average out the weighted sum of each cluster based on the number of datapoints assigned to it
#pragma omp parallel for schedule(static), collapse(2)
    for (int i = 0; i < clusters->numRows; i++)
    {
        for (int j = 0; j < clusters->numCols; j++)
        {
            clusters->data.at(i * clusters->numCols + j) /= clusterWeights->at(i);
        }
    }
}

int OMPLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

#pragma omp parallel for shared(distances, distanceFunc), schedule(static), reduction(+ \
                                                                                      : changed)
    for (int i = 0; i < matrix->numRows; i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = clustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != clustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

inline void OMPLloyd::updateClustering(const int &dataIdx, const int &clusterIdx)
{
    int &clusterAssignment = clustering->at(dataIdx);

    // only go through this update if the cluster assignment is going to change
    if (clusterAssignment != clusterIdx)
    {
        // cluster assignments are initialized to -1, so ignore decrement if datapoint has yet to be assigned
        if (clusterAssignment >= 0 && clusterWeights->at(clusterAssignment) > 0)
#pragma omp atomic
            clusterWeights->at(clusterAssignment) -= weights->at(dataIdx);
#pragma omp atomic
        clusterWeights->at(clusterIdx) += weights->at(dataIdx);
        clusterAssignment = clusterIdx;
    }
}

int OMPOptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

#pragma omp parallel for shared(distances, distanceFunc), schedule(static), reduction(+ \
                                                                                      : changed)
    for (int i = 0; i < matrix->numRows; i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        value_t dist = std::pow((*distanceFunc)(&*matrix->at(i), &*clusters->at(clustering->at(i)), matrix->numCols), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            // find closest cluster for each datapoint and update cluster assignment
            int before = clustering->at(i);
            auto closestCluster = findClosestCluster(i, distanceFunc);
            updateClustering(i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);

            // check if cluster assignments have changed
            if (before != clustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            distances->at(i) = dist;
        }
    }

    return changed;
}