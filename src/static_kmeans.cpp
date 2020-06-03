#include <boost/timer/timer.hpp>
#include <kmeans/filesystem/reader.hpp>
#include <kmeans/filesystem/writer.hpp>
#include <kmeans/kmeans.hpp>

using namespace hpkmeans;
typedef double value_t;
// constexpr char kmeansMethod[] = KPP;
constexpr Parallelism parallelism = Parallelism::OMP;
constexpr int numData             = 200000;
constexpr int dims                = 2;
constexpr int numClusters         = 30;
constexpr int repeats             = 100;

int main(int argc, char* argv[])
{
    std::string filepath = "/Users/ericdang/Documents/High_Performance_Computing_Fall_2019/K-Means/data/test_" +
                           std::to_string(numData) + "_" + std::to_string(dims) + ".txt";

    MatrixReader<value_t, parallelism> reader;
    ClusterResultWriter<value_t> writer(parallelism);
    auto data = reader.read(filepath, numData, dims);

    std::cout << "Method: " << KPP << "\nParallelism: " << parallelismToString(parallelism) << "\nData: " << filepath
              << '\n';

    KMeans<value_t, parallelism> kmeans(OPTKPP, OPTLLOYD);
    const Clusters<value_t>* results;
    {
        boost::timer::auto_cpu_timer t;
        results = kmeans.fit(&data, numClusters, repeats);
    }

    std::cout << "Error: " << results->getError() << '\n';
    // writer.writeClusterResults(results, 0, filepath);
}