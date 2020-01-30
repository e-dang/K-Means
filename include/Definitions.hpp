#pragma once

#include <vector>

typedef float value_t;

struct MPIDataChunks
{
    int rank;
    Matrix matrixChunk;
    std::vector<int> lengths;
    std::vector<int> displacements;
};

struct BundledAlgorithmData
{
    Matrix *matrix;
    std::vector<value_t> *weights;

    BundledAlgorithmData(Matrix *matrix, std::vector<value_t> *weights) : matrix(matrix), weights(weights){};
    virtual ~BundledAlgorithmData(){};
};

struct BundledMPIAlgorithmData : public BundledAlgorithmData
{
    MPIDataChunks *dataChunks;

    BundledMPIAlgorithmData(Matrix *matrix, std::vector<value_t> *weights, MPIDataChunks *dataChunks) : BundledAlgorithmData(matrix, weights), dataChunks(dataChunks){};
    ~BundledMPIAlgorithmData(){};
};

typedef std::vector<value_t> datapoint_t;
typedef std::vector<datapoint_t> dataset_t;
typedef std::vector<int> clustering_t;

typedef struct
{
    dataset_t data;
    std::vector<value_t> weights;
} coreset_t;

typedef struct
{
    int count;
    datapoint_t coords;
} cluster_t;

typedef std::vector<cluster_t> clusters_t;