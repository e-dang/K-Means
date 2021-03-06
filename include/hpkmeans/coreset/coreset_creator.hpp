#pragma once

#include <hpkmeans/coreset/coreset_creator_impl.hpp>
#include <hpkmeans/coreset/distributed_impl.hpp>
#include <hpkmeans/coreset/shared_mem_impl.hpp>
namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class CoresetCreator
{
public:
    CoresetCreator() :
        p_impl(std::make_unique<
               std::conditional_t<isSharedMemory(Level), SharedMemoryCoresetCreatorImpl<T, Level, DistanceFunc>,
                                  DistributedCoresetCreatorImpl<T, Level, DistanceFunc>>>())
    {
    }

    inline Coreset<T> createCoreset(const Matrix<T>* const data, const int32_t& sampleSize)
    {
        return p_impl->createCoreset(data, sampleSize);
    }

private:
    std::unique_ptr<AbstractCoresetCreatorImpl<T, Level, DistanceFunc>> p_impl;
};
}  // namespace hpkmeans