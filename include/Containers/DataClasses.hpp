#pragma once

#include <memory>

#include "Containers/Matrix.hpp"
#include "Utils/DistanceFunctors.hpp"

/**
 * @brief Class to wrap the clustering data generated by Kmeans.
 */
struct ClusterData
{
    // Public member variables
    Matrix mClusters;                      // the cluster centers
    std::vector<int32_t> mClustering;      // the cluster assignments of each datapoint
    std::vector<value_t> mClusterWeights;  // the sum of the weights of each datapoint assigned to a cluster

    /**
     * @brief Default constructor.
     */
    ClusterData(){};

    /**
     * @brief Construct a new ClusterData object.
     *
     * @param numData - The number of datapoints that are being clustered.
     * @param numFeatures - The number of features each datapoint has.
     * @param numClusters - The number of clusters that the data is being clustered into.
     */
    ClusterData(const int32_t& numData, const int32_t& numFeatures, const int32_t& numClusters) :
        mClusters(numClusters, numFeatures)
    {
        mClustering     = std::vector<int32_t>(numData, -1);
        mClusterWeights = std::vector<value_t>(numClusters, 0);
    }

    /**
     * @brief Destroy the ClusterData object
     */
    ~ClusterData(){};

    /**
     * @brief Overloaded assignment operator.
     *
     * @param lhs - An instance of ClusterData that is to be copied into the calling instance of ClusterData.
     */
    void operator=(const ClusterData& lhs)
    {
        mClusters       = lhs.mClusters;
        mClustering     = std::move(lhs.mClustering);
        mClusterWeights = std::move(lhs.mClusterWeights);
    }
};

struct KmeansData
{
    const int mRank;
    const int32_t mTotalNumData;
    const std::vector<int32_t> mLengths;
    const std::vector<int32_t> mDisplacements;
    const int32_t mDisplacement;

    const Matrix* const pData;
    const std::vector<value_t>* const pWeights;
    const std::shared_ptr<IDistanceFunctor> pDistanceFunc;

    // dynamic data that changes each repeat
    Matrix* pClusters;
    std::vector<int32_t>* pClustering;
    std::vector<value_t>* pClusterWeights;
    std::vector<value_t>* pSqDistances;

    KmeansData(const Matrix* const data, const std::vector<value_t>* const weights,
               std::shared_ptr<IDistanceFunctor> distanceFunc, const int& rank, const int32_t& totalNumData,
               const std::vector<int32_t> lengths, const std::vector<int32_t> displacements) :
        mRank(rank),
        mTotalNumData(totalNumData),
        mLengths(lengths),
        mDisplacements(displacements),
        mDisplacement(displacements.at(mRank)),
        pData(data),
        pWeights(weights),
        pDistanceFunc(distanceFunc){};

    void setClusterData(ClusterData* const clusterData)
    {
        pClusters       = &clusterData->mClusters;
        pClustering     = &clusterData->mClustering;
        pClusterWeights = &clusterData->mClusterWeights;
    }

    void setSqDistances(std::vector<value_t>* const sqDistances) { pSqDistances = sqDistances; }

    int32_t& clusteringAt(const int32_t& dataIdx) { return pClustering->at(mDisplacement + dataIdx); }
    value_t& sqDistancesAt(const int32_t& dataIdx) { return pSqDistances->at(mDisplacement + dataIdx); }
    value_t& clusterWeightsAt(const int32_t& clusterIdx) { return pClusterWeights->at(clusterIdx); }
};

/**
 * @brief A return structure that couples the distance between a point and its closest cluster and the index of that
 *        cluster together.
 */
struct ClosestCluster
{
    // Public member variables
    int32_t clusterIdx;
    double distance;
};

struct ClusterResults
{
    value_t mError;
    ClusterData mClusterData;
    std::vector<value_t> mSqDistances;

    ClusterResults() : mError(-1) {}

    ~ClusterResults() {}
};

struct Coreset
{
    Matrix data;
    std::vector<value_t> weights;
};

struct MPIData
{
    int rank;
    int numProcs;
    std::vector<int32_t> lengths;
    std::vector<int32_t> displacements;
};

enum Initializer
{
    InitNull = 0,
    KPP      = 1 << 0,
    OptKPP   = 1 << 1
};

enum Maximizer
{
    MaxNull  = 0,
    Lloyd    = 1 << 2,
    OptLloyd = 1 << 3
};

enum CoresetCreator
{
    None      = 0,
    LWCoreset = 1 << 4,
};

enum Parallelism
{
    ParaNull = 0,
    Serial   = 1 << 5,
    OMP      = 1 << 6,
    MPI      = 1 << 7,
    Hybrid   = 1 << 8
};

enum Variant
{
    Reg,
    Opt,
    SpecificCoreset
};