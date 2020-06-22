#include <boost/timer/timer.hpp>
#include <hpkmeans/filesystem/reader.hpp>
#include <hpkmeans/filesystem/writer.hpp>
#include <hpkmeans/kmeans.hpp>

using namespace hpkmeans;
typedef double value_t;
constexpr Parallelism parallelism = Parallelism::Hybrid;
constexpr bool coreset            = true;
constexpr int numData             = 200000;
constexpr int dims                = 2;
constexpr int numClusters         = 30;
constexpr int repeats             = 10;
constexpr int coresetRepeats      = 10;
constexpr int numIters            = 1;
constexpr int sampleSize          = numData / 10;

std::unique_ptr<KMeans<value_t, parallelism>> createKmeans()
{
    if constexpr (coreset)
        return std::make_unique<KMeans<value_t, parallelism>>(OPTKPP, OPTLLOYD, coresetRepeats, sampleSize);
    else
        return std::make_unique<KMeans<value_t, parallelism>>(OPTKPP, OPTLLOYD);
}

void runSharedMemory(std::string& filepath)
{
    MatrixReader<value_t, parallelism> reader;
    ClusterResultWriter<value_t, parallelism> writer(parallelism);

    auto data = reader.read(filepath, numData, dims);

    auto kmeans = createKmeans();
    const Clusters<value_t, parallelism>* results;
    for (int i = 0; i < numIters; ++i)
    {
        boost::timer::auto_cpu_timer t;
        results = kmeans->fit(&data, numClusters, repeats);
    }

    std::cout << "Error: " << results->getError() << '\n';
    writer.writeClusterResults(results, 0, filepath);
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
    for (int i = 0; i < numIters; ++i)
    {
        boost::timer::auto_cpu_timer t;
        results = kmeans->fit(&data, numClusters, repeats);
    }

    if (rank == 0)
    {
        std::cout << "Error: " << results->getError() << '\n';
        writer.writeClusterResults(results, 0, filepath);
    }

    MPI_Finalize();
}

int main(int argc, char* argv[])
{
    std::string filepath = /*INSERT FILEPATH HERE*/ std::to_string(numData) + "_" + std::to_string(dims) + ".txt";

    std::cout << "Method: " << KPP << "\nParallelism: " << parallelismToString(parallelism) << "\nData: " << filepath
              << '\n';

    if (isSharedMemory(parallelism))
        runSharedMemory(filepath);
    else
        runDistributed(filepath);
}