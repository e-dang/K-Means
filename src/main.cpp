#include <chrono>
#include <iostream>

#include "Containers/Definitions.hpp"
#include "Kmeans/KmeansFacade.hpp"
#include "Utils/DistanceFunctors.hpp"
#include "Utils/Reader.hpp"
#include "Utils/Writer.hpp"
#include "mpi.h"
#include "sarge/sarge.h"

const int DEFAULT_REPEATS = 10;
using namespace HPKmeans;
void parseFilePath(Sarge& sarge, std::string& filepath)
{
    if (!sarge.getTextArgument(0, filepath))
    {
        std::cerr << "Must specify a path to a datafile!" << std::endl;
        exit(1);
    }
}

int32_t parseNumData(Sarge& sarge)
{
    std::string rows;
    int32_t numData;
    if (sarge.getFlag("rows", rows))
    {
        numData = std::atoi(rows.c_str());
        if (numData <= 0)
        {
            std::cerr << "The number of data points must be greater than 0!" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cerr << "Must specify the number of datapoints to be clustered!" << std::endl;
        exit(1);
    }

    return numData;
}

int32_t parseNumFeatures(Sarge& sarge)
{
    std::string cols;
    int32_t numFeatures;
    if (sarge.getFlag("cols", cols))
    {
        numFeatures = std::atoi(cols.c_str());
        if (numFeatures <= 0)
        {
            std::cerr << "The number of features in each data point must be greater than 0!" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cerr << "Must specify the number of features in each data point!" << std::endl;
        exit(1);
    }

    return numFeatures;
}

int32_t parseNumClusters(Sarge& sarge)
{
    std::string clusters;
    int32_t numClusters;
    if (sarge.getFlag("clusters", clusters))
    {
        numClusters = std::atoi(clusters.c_str());
        if (numClusters <= 0)
        {
            std::cerr << "The number of clusters must be greater than 0!" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cerr << "Must specify the number of clusters to cluster the data into!" << std::endl;
        exit(1);
    }

    return numClusters;
}

int32_t parseSampleSize(Sarge& sarge)
{
    std::string samples;
    int32_t sampleSize;
    if (sarge.getFlag("samples", samples))
    {
        sampleSize = std::atoi(samples.c_str());
        if (sampleSize <= 0)
        {
            std::cerr << "The sample size must be greater than 0!" << std::endl;
            exit(1);
        }

        return sampleSize;
    }

    return -1;
}

int parseNumRestarts(Sarge& sarge)
{
    std::string restarts;
    int numRestarts;
    if (sarge.getFlag("restarts", restarts))
    {
        numRestarts = std::atoi(restarts.c_str());
        if (numRestarts <= 0)
        {
            std::cerr << "The number of restarts must be greater than 0!" << std::endl;
            exit(1);
        }

        return numRestarts;
    }

    return DEFAULT_REPEATS;
}

Initializer parseInitializer(Sarge& sarge)
{
    Initializer initializer = InitNull;
    bool argPresent;
    std::string kpp;
    if (sarge.getFlag("kpp", kpp))
        initializer = KPP;

    std::string optkpp;
    argPresent = sarge.getFlag("optkpp", optkpp);
    if (argPresent && initializer == InitNull)
    {
        initializer = OptKPP;
    }
    else if (argPresent && initializer != InitNull)
    {
        std::cerr << "Cannot specify two types of initializer methods!" << std::endl;
        exit(1);
    }
    else if (!argPresent && initializer == InitNull)
    {
        std::cerr << "Must specify an initializer method!" << std::endl;
        exit(1);
    }

    return initializer;
}

Maximizer parseMaximizer(Sarge& sarge)
{
    Maximizer maximizer = MaxNull;
    bool argPresent;
    std::string lloyd;
    if (sarge.getFlag("lloyd", lloyd))
        maximizer = Lloyd;

    std::string optlloyd;
    argPresent = sarge.getFlag("optlloyd", optlloyd);
    if (sarge.getFlag("optlloyd", optlloyd) && maximizer == MaxNull)
    {
        maximizer = OptLloyd;
    }
    else if (argPresent && maximizer != MaxNull)
    {
        std::cerr << "Cannot specify two types of maximizer methods!" << std::endl;
        exit(1);
    }
    else if (!argPresent && maximizer == MaxNull)
    {
        std::cerr << "Must specify a maximizer method!" << std::endl;
        exit(1);
    }

    return maximizer;
}

CoresetCreator parseCoresetCreator(Sarge& sarge, int32_t& sampleSize)
{
    CoresetCreator coresetCreator = None;
    std::string coreset;
    if (sarge.getFlag("coreset", coreset))
    {
        coresetCreator = LWCoreset;
        if (sampleSize <= 0)
        {
            std::cerr << "Must give a sample size greater than 0 when using a coreset!" << std::endl;
            exit(1);
        }
    }

    return coresetCreator;
}

Parallelism parseParallelism(Sarge& sarge)
{
    Parallelism parallelism = ParaNull;
    bool argPresent;
    std::string serial;
    if (sarge.getFlag("serial", serial))
        parallelism = Serial;

    std::string omp;
    argPresent = sarge.getFlag("omp", omp);
    if (argPresent && parallelism == ParaNull)
    {
        parallelism = OMP;
    }
    else if (argPresent && parallelism != ParaNull)
    {
        std::cerr << "Cannot specify two different levels of parallelism!" << std::endl;
        exit(1);
    }

    std::string mpi;
    argPresent = sarge.getFlag("mpi", mpi);
    if (argPresent && parallelism == ParaNull)
    {
        parallelism = MPI;
    }
    else if (argPresent && parallelism != ParaNull)
    {
        std::cerr << "Cannot specify two different levels of parallelism!" << std::endl;
        exit(1);
    }

    std::string hybrid;
    argPresent = sarge.getFlag("hybrid", hybrid);
    if (argPresent && parallelism == ParaNull)
    {
        parallelism = Hybrid;
    }
    else if (argPresent && parallelism != ParaNull)
    {
        std::cerr << "Cannot specify two different levels of parallelism!" << std::endl;
        exit(1);
    }
    else if (!argPresent && parallelism == ParaNull)
    {
        std::cerr << "Must specify a level of parallelism!" << std::endl;
        exit(1);
    }

    return parallelism;
}

void runDistributed(int& argc, char** argv, std::string filepath, const int32_t& numData, const int32_t& numFeatures,
                    const int32_t& numClusters, const int32_t& sampleSize, const int& numRestarts,
                    Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator,
                    Parallelism parallelism)
{
    int rank = 0, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    MPIMatrixReader<double> reader;
    auto data = reader.read(filepath, numData, numFeatures);

    Kmeans<double> kmeans(initializer, maximizer, coresetCreator, parallelism,
                          std::make_shared<EuclideanDistance<double>>(), sampleSize);

    auto start          = std::chrono::high_resolution_clock::now();
    auto clusterResults = kmeans.fit(&data, numClusters, numRestarts);
    auto stop           = std::chrono::high_resolution_clock::now();
    auto duration       = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    if (rank == 0)
    {
        ClusterResultWriter<double> writer(initializer, maximizer, coresetCreator, parallelism);
        writer.writeClusterResults(clusterResults, duration, filepath);
    }

    MPI_Finalize();
}

void runSharedMemory(int& argc, char** argv, std::string filepath, const int32_t& numData, const int32_t& numFeatures,
                     const int32_t& numClusters, const int32_t& sampleSize, const int& numRestarts,
                     Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator,
                     Parallelism parallelism)
{
    MatrixReader<double> reader;
    auto data = reader.read(filepath, numData, numFeatures);

    Kmeans<double> kmeans(initializer, maximizer, coresetCreator, parallelism,
                          std::make_shared<EuclideanDistance<double>>(), sampleSize);

    auto start          = std::chrono::high_resolution_clock::now();
    auto clusterResults = kmeans.fit(&data, numClusters, numRestarts);
    auto stop           = std::chrono::high_resolution_clock::now();
    auto duration       = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    ClusterResultWriter<double> writer(initializer, maximizer, coresetCreator, parallelism);
    writer.writeClusterResults(clusterResults, duration, filepath);
}

int main(int argc, char** argv)
{
    Sarge sarge;
    sarge.setArgument("f", "file", "The path the data file.", false);
    sarge.setArgument("r", "rows", "The number of data points (rows) in the matrix.", true);
    sarge.setArgument("c", "cols", "The number features (columns) in each data point.", true);
    sarge.setArgument("k", "clusters", "The number of clusters to cluster the data into.", true);
    sarge.setArgument("t", "restarts", "The number of tries to find the best clustering. Defaults to 10.", true);
    sarge.setArgument("s", "samples", "The sample size to use for a Coreset.", true);
    sarge.setArgument("", "kpp", "A switch to use K++ for initialization.", false);
    sarge.setArgument("", "optkpp", "A switch to use OptimizedK++ for initialization.", false);
    sarge.setArgument("", "lloyd", "A switch to use Lloyd for maximization.", false);
    sarge.setArgument("", "optlloyd", "A switch to use OptimizedLloyd for maximization.", false);
    sarge.setArgument("", "coreset", "A switch to use a Coreset for quicker approximation.", false);
    sarge.setArgument("", "serial", "A switch to toggle serialized execution.", false);
    sarge.setArgument("", "omp", "A switch to toggle OMP execution.", false);
    sarge.setArgument("", "mpi", "A switch to toggle MPI execution.", false);
    sarge.setArgument("", "hybrid", "A switch to toggle Hybrid (OMP and MPI) execution.", false);

    if (!sarge.parseArguments(argc, argv))
    {
        std::cerr << "Failed to parse arguments." << std::endl;
        exit(1);
    }

    std::string filepath;
    parseFilePath(sarge, filepath);
    auto numData        = parseNumData(sarge);
    auto numFeatures    = parseNumFeatures(sarge);
    auto numClusters    = parseNumClusters(sarge);
    auto sampleSize     = parseSampleSize(sarge);
    auto numRestarts    = parseNumRestarts(sarge);
    auto initializer    = parseInitializer(sarge);
    auto maximizer      = parseMaximizer(sarge);
    auto coresetCreator = parseCoresetCreator(sarge, sampleSize);
    auto parallelism    = parseParallelism(sarge);

    if (parallelism == MPI || parallelism == Hybrid)
    {
        runDistributed(argc, argv, filepath, numData, numFeatures, numClusters, sampleSize, numRestarts, initializer,
                       maximizer, coresetCreator, parallelism);
    }
    else
    {
        runSharedMemory(argc, argv, filepath, numData, numFeatures, numClusters, sampleSize, numRestarts, initializer,
                        maximizer, coresetCreator, parallelism);
    }

    exit(0);
}
