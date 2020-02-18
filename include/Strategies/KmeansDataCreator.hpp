#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Utils/Utils.hpp"
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class IKmeansDataCreator
{
public:
    virtual ~IKmeansDataCreator() = default;

    virtual KmeansData<precision, int_size> create(const Matrix<precision, int_size>* const data,
                                                   const std::vector<precision>* const weights,
                                                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SharedMemoryKmeansDataCreator : public IKmeansDataCreator<precision, int_size>
{
public:
    ~SharedMemoryKmeansDataCreator() = default;

    KmeansData<precision, int_size> create(const Matrix<precision, int_size>* const data,
                                           const std::vector<precision>* const weights,
                                           std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override;
};

template <typename precision = double, typename int_size = int32_t>
class MPIKmeansDataCreator : public IKmeansDataCreator<precision, int_size>
{
public:
    ~MPIKmeansDataCreator() = default;

    KmeansData<precision, int_size> create(const Matrix<precision, int_size>* const data,
                                           const std::vector<precision>* const weights,
                                           std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override;
};

template <typename precision, typename int_size>
KmeansData<precision, int_size> SharedMemoryKmeansDataCreator<precision, int_size>::create(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc)
{
    return KmeansData<precision, int_size>(data, weights, distanceFunc, 0, data->size(),
                                           std::vector<int_size>(1, data->size()), std::vector<int_size>(1, 0));
}

template <typename precision, typename int_size>
KmeansData<precision, int_size> MPIKmeansDataCreator<precision, int_size>::create(
  const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc)
{
    auto totalNumData = getTotalNumDataMPI(data);
    auto mpiData      = getMPIData(totalNumData);
    return KmeansData<precision, int_size>(data, weights, distanceFunc, mpiData.rank, totalNumData, mpiData.lengths,
                                           mpiData.displacements);
}
}  // namespace HPKmeans