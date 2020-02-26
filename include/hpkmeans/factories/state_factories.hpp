#pragma once

#include <hpkmeans/data_types/enums.hpp>
#include <hpkmeans/data_types/kmeans_state.hpp>
#include <hpkmeans/data_types/matrix.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
class IKmeansStateFactory
{
public:
    virtual ~IKmeansStateFactory() = default;

    virtual KmeansState<precision, int_size>* createState(
      const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
      std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) = 0;
};

template <typename precision, typename int_size>
class SharedMemoryKmeansStateFactory : public IKmeansStateFactory<precision, int_size>
{
public:
    ~SharedMemoryKmeansStateFactory() = default;

    KmeansState<precision, int_size>* createState(const Matrix<precision, int_size>* const data,
                                                  const std::vector<precision>* const weights,
                                                  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override
    {
        return new KmeansState<precision, int_size>(data, weights, distanceFunc,
                                                    new SharedMemoryDataChunks<int_size>(data->size()));
    }
};

template <typename precision, typename int_size>
class MPIKmeansStateFactory : public IKmeansStateFactory<precision, int_size>
{
public:
    ~MPIKmeansStateFactory() = default;

    KmeansState<precision, int_size>* createState(const Matrix<precision, int_size>* const data,
                                                  const std::vector<precision>* const weights,
                                                  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override
    {
        return new KmeansState<precision, int_size>(data, weights, distanceFunc,
                                                    new MPIDataChunks<int_size>(data->size()));
    }
};

template <typename precision, typename int_size>
class KmeansStateAbstractFactory
{
public:
    IKmeansStateFactory<precision, int_size>* createStateFactory(Parallelism parallelism)
    {
        switch (parallelism)
        {
            case (Serial):
                return new SharedMemoryKmeansStateFactory<precision, int_size>();
            case (OMP):
                return new SharedMemoryKmeansStateFactory<precision, int_size>();
            case (MPI):
                return new MPIKmeansStateFactory<precision, int_size>();
            case (Hybrid):
                return new MPIKmeansStateFactory<precision, int_size>();
            default:
                std::cerr << "Invalid parallelism specifier provided." << std::endl;
                exit(1);
        }
    }
};
}  // namespace HPKmeans