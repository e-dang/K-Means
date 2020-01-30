#include "AbstractKmeansAlgorithms.hpp"
#include "mpi.h"

// void AbstractMPIKmeansAlgorithm::bcastClusterData()
// {
//     MPI_Bcast(pClustering->data(), pClustering->size(), MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(pClusters->data.data(), pClusters->data.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
// }