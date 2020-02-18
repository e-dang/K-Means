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
    virtual KmeansData<precision, int_size> create(const Matrix<precision, int_size>* const data,
                                                   const std::vector<precision>* const weights,
                                                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SharedMemoryKmeansDataCreator : public IKmeansDataCreator<precision, int_size>
{
public:
    KmeansData<precision, int_size> create(const Matrix<precision, int_size>* const data,
                                           const std::vector<precision>* const weights,
                                           std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override
    {
        return KmeansData<precision, int_size>(data, weights, distanceFunc, 0, data->size(),
                                               std::vector<int_size>(1, data->size()), std::vector<int_size>(1, 0));
    }
};

template <typename precision = double, typename int_size = int32_t>
class MPIKmeansDataCreator : public IKmeansDataCreator<precision, int_size>
{
public:
    KmeansData<precision, int_size> create(const Matrix<precision, int_size>* const data,
                                           const std::vector<precision>* const weights,
                                           std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) override
    {
        auto totalNumData = getTotalNumDataMPI(data);
        auto mpiData      = getMPIData(totalNumData);
        return KmeansData<precision, int_size>(data, weights, distanceFunc, mpiData.rank, totalNumData, mpiData.lengths,
                                               mpiData.displacements);
    }
};
}  // namespace HPKmeans