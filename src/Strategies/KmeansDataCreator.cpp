#include "Strategies/KmeansDataCreator.hpp"

#include "Utils/Utils.hpp"

KmeansData SharedMemoryKmeansDataCreator::create(const Matrix* const data, const std::vector<value_t>* const weights,
                                                 std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    return KmeansData(data, weights, distanceFunc, 0, data->getNumData(),
                      std::vector<int32_t>(1, data->getNumData()), std::vector<int32_t>(1, 0));
}

KmeansData MPIKmeansDataCreator::create(const Matrix* const data, const std::vector<value_t>* const weights,
                                        std::shared_ptr<IDistanceFunctor> distanceFunc)
{
    auto totalNumData = getTotalNumDataMPI(data);
    auto mpiData      = getMPIData(totalNumData);
    return KmeansData(data, weights, distanceFunc, mpiData.rank, totalNumData, mpiData.lengths, mpiData.displacements);
}