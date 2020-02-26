#pragma once

#include <hpkmeans/data_types/enums.hpp>
#include <hpkmeans/data_types/matrix.hpp>
#include <hpkmeans/factories/kmeans_factory.hpp>
#include <hpkmeans/kmeans/kmeans_wrapper.hpp>

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class Kmeans
{
protected:
    std::unique_ptr<AbstractKmeansWrapper<precision, int_size>> pKmeans;
    std::unique_ptr<KmeansFactory<precision, int_size>> pFactory;

    Initializer mInitializer;
    Maximizer mMaximizer;
    CoresetCreator mCoreset;
    Parallelism mParallelism;
    int_size mSampleSize;

public:
    Kmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
           const Parallelism parallelism, std::shared_ptr<IDistanceFunctor<precision>> distanceFunc,
           const int_size sampleSize = -1);

    ~Kmeans() = default;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts);

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             std::vector<precision>* weights);

    bool setKmeans(const Initializer initializer, const Maximizer maximizer, const CoresetCreator coreset,
                   const Parallelism parallelism, std::shared_ptr<IDistanceFunctor<precision>> distanceFunc,
                   const int_size sampleSize = -1);

    void setDistanceFunc(IDistanceFunctor<precision>* distanceFunc) { pKmeans->setDistanceFunc(distanceFunc); }

private:
    void setInitializer(const Initializer initializer) { mInitializer = initializer; }

    void setMaximizer(const Maximizer maximizer) { mMaximizer = maximizer; }

    void setCoresetCreator(const CoresetCreator coreset) { mCoreset = coreset; }

    void setParallelism(const Parallelism parallelism) { mParallelism = parallelism; }

    bool sampleSizeCheck();

    bool isValidSampleSize(const Matrix<precision, int_size>* const data);
};

template <typename precision, typename int_size>
Kmeans<precision, int_size>::Kmeans(const Initializer initializer, const Maximizer maximizer,
                                    const CoresetCreator coreset, const Parallelism parallelism,
                                    std::shared_ptr<IDistanceFunctor<precision>> distanceFunc,
                                    const int_size sampleSize) :
    pKmeans(nullptr),
    pFactory(new KmeansFactory<precision, int_size>()),
    mInitializer(initializer),
    mMaximizer(maximizer),
    mCoreset(coreset),
    mParallelism(parallelism),
    mSampleSize(sampleSize)
{
    setKmeans(initializer, maximizer, coreset, parallelism, distanceFunc, sampleSize);
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> Kmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts)
{
    if (isValidSampleSize(data) && pKmeans != nullptr)
        return pKmeans->fit(data, numClusters, numRestarts);

    return nullptr;
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> Kmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts,
  std::vector<precision>* weights)
{
    if (isValidSampleSize(data) && pKmeans != nullptr)
        return pKmeans->fit(data, numClusters, numRestarts, weights);

    return nullptr;
}

template <typename precision, typename int_size>
bool Kmeans<precision, int_size>::setKmeans(const Initializer initializer, const Maximizer maximizer,
                                            const CoresetCreator coreset, const Parallelism parallelism,
                                            std::shared_ptr<IDistanceFunctor<precision>> distanceFunc,
                                            const int_size sampleSize)
{
    if (sampleSizeCheck())
    {
        setInitializer(initializer);
        setMaximizer(maximizer);
        setCoresetCreator(coreset);
        setParallelism(parallelism);
        pKmeans = std::unique_ptr<AbstractKmeansWrapper<precision, int_size>>(
          pFactory->createKmeans(initializer, maximizer, coreset, parallelism, distanceFunc, sampleSize));

        return true;
    }

    return false;
}

template <typename precision, typename int_size>
bool Kmeans<precision, int_size>::sampleSizeCheck()
{
    if (mCoreset == None || (mCoreset != None && mSampleSize >= 0))
        return true;

    std::cout << "A sample size must be given when using CoresetKmeans." << std::endl;
    return false;
}

template <typename precision, typename int_size>
bool Kmeans<precision, int_size>::isValidSampleSize(const Matrix<precision, int_size>* const data)
{
    if (mParallelism == MPI || mParallelism == Hybrid)
    {
        MPI_Datatype mpi_int_size;
        MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(int_size), &mpi_int_size);
        if (mSampleSize <= MPIDataChunks<int_size>::getTotalNumData(data->size(), mpi_int_size))
            return true;
    }
    else
    {
        if (mSampleSize <= data->rows())
            return true;
    }

    return false;
}
}  // namespace HPKmeans