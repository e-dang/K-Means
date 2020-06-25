#include <boost/timer/timer.hpp>
#include <hpkmeans/filesystem/reader.hpp>
#include <hpkmeans/filesystem/writer.hpp>
#include <hpkmeans/kmeans.hpp>

using namespace hpkmeans;

constexpr bool strings_equal(char const* a, char const* b)
{
    return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

constexpr Parallelism getParallelism()
{
    if constexpr (strings_equal(PARALLELISM, "serial"))
        return Parallelism::Serial;
    else if constexpr (strings_equal(PARALLELISM, "omp"))
        return Parallelism::OMP;
    else if constexpr (strings_equal(PARALLELISM, "mpi"))
        return Parallelism::MPI;
    else
        return Parallelism::Hybrid;
}

typedef double value_t;
constexpr char lloydMethod[]      = METHOD;
constexpr Parallelism parallelism = getParallelism();
constexpr bool coreset            = CORESET;
constexpr int numData             = 1000000;
constexpr int dims                = 10;
constexpr int numClusters         = 50;
constexpr int repeats             = 10;
constexpr int coresetRepeats      = 1;
constexpr int numIters            = 1;
constexpr int sampleSize          = numData / 10;

std::unique_ptr<KMeans<value_t, parallelism>> createKmeans()
{
    if constexpr (coreset)
        return std::make_unique<KMeans<value_t, parallelism>>(OPTKPP, lloydMethod, coresetRepeats, sampleSize);
    else
        return std::make_unique<KMeans<value_t, parallelism>>(OPTKPP, lloydMethod);
}

void runSharedMemory(std::string& filepath)
{
    MatrixReader<value_t, parallelism> reader;
    ClusterResultWriter<value_t, parallelism> writer(parallelism);

    auto data = reader.read(filepath, numData, dims);

    auto kmeans = createKmeans();
    const Clusters<value_t, parallelism>* results;
    int64_t time;
    for (int i = 0; i < numIters; ++i)
    {
        boost::timer::auto_cpu_timer t;
        results = kmeans->fit(&data, numClusters, repeats);
        time    = t.elapsed().wall;
    }

    std::cout << "Error: " << results->getError() << '\n';
    writer.writeClusterResults(results, time, filepath);
}

void runDistributed(std::string& filepath)
{
    int rank;
    int size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MatrixReader<value_t, parallelism> reader;
    ClusterResultWriter<value_t, parallelism> writer(parallelism);

    auto data = reader.read(filepath, numData, dims);

    auto kmeans = createKmeans();
    const Clusters<value_t, parallelism>* results;
    int64_t time;
    for (int i = 0; i < numIters; ++i)
    {
        boost::timer::auto_cpu_timer t;
        results = kmeans->fit(&data, numClusters, repeats);
        time    = t.elapsed().wall;
    }

    if (rank == 0)
    {
        std::cout << "Error: " << results->getError() << '\n';
        writer.writeClusterResults(results, time, filepath);
    }

    MPI_Finalize();
}

int main(int argc, char* argv[])
{
    std::string filepath = /*INSERT FILEPATH HERE*/ std::to_string(numData) + "_" + std::to_string(dims) + "_" +
                           std::to_string(numClusters) + ".txt";

    std::cout << "Method: " << OPTKPP << "\nParallelism: " << parallelismToString(parallelism)
              << "\nCoreset: " << coreset << "\nData: " << filepath << '\n';

    if (isSharedMemory(parallelism))
        runSharedMemory(filepath);
    else
        runDistributed(filepath);
}