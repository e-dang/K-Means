#pragma once

#include <vector>

typedef boost::mt19937 RNGType;

typedef float value_t;
typedef std::vector<value_t> datapoint_t;
typedef std::vector<datapoint_t> dataset_t;
typedef std::vector<int> clustering_t;

typedef struct
{
    int count;
    datapoint_t coords;
} cluster_t;

typedef std::vector<cluster_t> clusters_t;