#pragma once

#include <iostream>
#include <vector>

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Wrapper class around std::vector<value_t> that helps manipulate the vector as if it were a 2D nested vector.
 *        Keeping the data as a 1D vector rather than a nested vector makes it easier to implement with MPI.
 */
class Matrix
{
private:
    // Member variables
    std::vector<value_t> mData;
    int mNumRows;  // this is the number of datapoints in the matrix
    int mNumCols;  // this is the number of features of each datapoint in the matrix

public:
    Matrix() {}

    Matrix(const int& numRows, const int& numCols)
    {
        mData.reserve(numRows * numCols);
        mNumRows = numRows;
        mNumCols = numCols;
    }

    Matrix(std::vector<value_t> data, const int& numRows, const int& numCols)
    {
        if (data.size() > numRows * numCols)
        {
            throw std::runtime_error("The vector has more data than was specified.");
        }
        mData = data;
        mData.reserve(numRows * numCols);
        mNumRows = numRows;
        mNumCols = numCols;
    }

    /**
     * @brief Destroy the Matrix object
     */
    ~Matrix(){};

    /**
     * @brief Function to access the beginning of a "row" of the matrix, where each row of the matrix is a datapoint.
     *        This function returns an iterator to the beginning of the row which can be used to access the rest of the
     *        row.
     *
     * @param row - The number of the row to retrieve.
     * @return value_t *
     */
    value_t* at(const int& row) { return mData.data() + (row * mNumCols); }

    /**
     * @brief Function to access a specific value in the matrix.
     *
     * @param row - The row number that the value is in.
     * @param col - The column number that the value is in.
     * @return value_t&
     */
    value_t& at(const int& row, const int& col) { return *(this->at(row) + col); }

    void appendDataPoint(value_t* datapoint)
    {
        if (getNumData() < mNumRows)
        {
            std::copy(datapoint, datapoint + mNumCols, std::back_inserter(mData));
        }
        else
        {
            throw std::overflow_error("Cannot append to full matrix");
        }
    }

    void fill(const int& val) { std::fill(mData.begin(), mData.end(), val); }

    void resize(const int& val) { mData.resize(val * mNumCols); }

    value_t* data() { return mData.data(); }

    int size() { return mData.size(); }
    int getNumData() { return mData.size() / mNumCols; }
    int getMaxNumData() { return mNumRows; }
    int getNumFeatures() { return mNumCols; }

    void operator=(const Matrix& lhs)
    {
        mData    = std::move(lhs.mData);
        mNumRows = lhs.mNumRows;
        mNumCols = lhs.mNumCols;
    }

    void display()
    {
        for (auto& val : mData)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};

/**
 * @brief Class to wrap the clustering data generated by Kmeans.
 */
struct ClusterData
{
    // Public member variables
    Matrix mClusters;                      // the cluster centers
    std::vector<int> mClustering;          // the cluster assignments of each datapoint
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
    ClusterData(const int& numData, const int& numFeatures, const int& numClusters) :
        mClusters(numClusters, numFeatures)
    {
        mClustering     = std::vector<int>(numData, -1);
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

/**
 * @brief A return structure that couples the distance between a point and its closest cluster and the index of that
 *        cluster together.
 */
struct ClosestCluster
{
    // Public member variables
    int clusterIdx;
    value_t distance;
};

struct StaticData
{
    // user data
    Matrix* pData;
    std::vector<value_t>* pWeights;
    IDistanceFunctor* pDistanceFunc;

    // chunk data
    int mRank;
    int mTotalNumData;
    std::vector<int> mLengths;
    std::vector<int> mDisplacements;
};

struct ClusterResults
{
    int mError;
    ClusterData mClusterData;
    std::vector<value_t> mSqDistances;

    ClusterResults() : mError(-1) {}

    ~ClusterResults() {}
};