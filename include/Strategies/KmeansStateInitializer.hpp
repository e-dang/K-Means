#pragma once

#include "Containers/KmeansState.hpp"
#include "Utils/Utils.hpp"

namespace HPKmeans
{
template <typename precision, typename int_size>
class IKmeansStateInitializer
{
public:
    virtual ~IKmeansStateInitializer() = default;

    virtual KmeansState<precision, int_size> initializeState(
      const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
      std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) = 0;
};

template <typename precision, typename int_size>
class SharedMemoryKmeansStateInitializer : public IKmeansStateInitializer<precision, int_size>
{
public:
    ~SharedMemoryKmeansStateInitializer() = default;

    KmeansState<precision, int_size> initializeState(
      const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
      std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override;
};

template <typename precision, typename int_size>
class MPIKmeansStateInitializer : public IKmeansStateInitializer<precision, int_size>
{
public:
    ~MPIKmeansStateInitializer() = default;

    KmeansState<precision, int_size> initializeState(
      const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
      std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override;
};

template <typename precision, typename int_size>
KmeansState<precision, int_size> SharedMemoryKmeansStateInitializer<precision, int_size>::initializeState(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc)
{
    return KmeansState<precision, int_size>(data, weights, distanceFunc, 0, 1, data->size(),
                                            std::vector<int_size>(1, data->size()), std::vector<int_size>(1, 0));
}

template <typename precision, typename int_size>
KmeansState<precision, int_size> MPIKmeansStateInitializer<precision, int_size>::initializeState(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc)
{
    auto totalNumData = getTotalNumDataMPI(data);
    auto mpiData      = getMPIData(totalNumData);
    return KmeansState<precision, int_size>(data, weights, distanceFunc, mpiData.rank, mpiData.numProcs, totalNumData,
                                            mpiData.lengths, mpiData.displacements);
}
}  // namespace HPKmeans