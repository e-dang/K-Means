#pragma once

#include "Containers/DataClasses.hpp"
#include "Strategies/Averager.hpp"
#include "Strategies/ClosestClusterFinder.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"
#include "Strategies/CoresetClusteringFinisher.hpp"
#include "Strategies/CoresetDistributionCalculator.hpp"
#include "Strategies/DistanceSumCalculator.hpp"
#include "Strategies/KmeansDataCreator.hpp"
#include "Strategies/PointReassigner.hpp"
#include "Strategies/RandomSelector.hpp"

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class AbstractStrategyFactory
{
public:
    virtual ~AbstractStrategyFactory() = default;

    IClosestClusterFinder<precision, int_size>* createClosestClusterFinder(Variant variant);

    AbstractClosestClusterUpdater<precision, int_size>* createClosestClusterUpdater(Variant variant);

    AbstractPointReassigner<precision, int_size>* createPointReassigner(Variant variant);

    IMultiWeightedRandomSelector<precision, int_size>* createMultiWeightedRandomSelector();

    IWeightedRandomSelector<precision, int_size>* createWeightedRandomSelector();

    AbstractClusteringDataUpdater<precision, int_size>* createCoresetClusteringDataUpdater();

    virtual AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() = 0;

    virtual AbstractWeightedAverager<precision, int_size>* createWeightedAverager() = 0;

    virtual AbstractAverager<precision, int_size>* createVectorAverager() = 0;

    virtual IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() = 0;

    virtual ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() = 0;

    virtual IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() = 0;

    virtual AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() = 0;

    virtual AbstractPointReassigner<precision, int_size>* createRegPointReassigner() = 0;

    virtual AbstractPointReassigner<precision, int_size>* createOptPointReassigner() = 0;
};

template <typename precision = double, typename int_size = int32_t>
class SerialStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    ~SerialStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override;

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override;

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override;

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override;

    AbstractAverager<precision, int_size>* createVectorAverager() override;

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override;

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override;

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override;

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override;
};

template <typename precision = double, typename int_size = int32_t>
class OMPStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    ~OMPStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override;

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override;

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override;

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override;

    AbstractAverager<precision, int_size>* createVectorAverager() override;

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override;

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override;

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override;

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override;
};

template <typename precision = double, typename int_size = int32_t>
class MPIStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    ~MPIStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override;

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override;

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override;

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override;

    AbstractAverager<precision, int_size>* createVectorAverager() override;

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override;

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override;

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override;

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override;
};

template <typename precision = double, typename int_size = int32_t>
class HybridStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    ~HybridStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override;

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override;

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override;

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override;

    AbstractAverager<precision, int_size>* createVectorAverager() override;

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override;

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override;

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override;

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override;
};

template <typename precision, typename int_size>
IClosestClusterFinder<precision, int_size>* AbstractStrategyFactory<precision, int_size>::createClosestClusterFinder(
  Variant variant)
{
    switch (variant)
    {
        case (Reg):
            return new ClosestClusterFinder<precision, int_size>();
        case (Opt):
            return new ClosestNewClusterFinder<precision, int_size>();
        default:
            throw std::runtime_error("Invalid ClosestClusterFinder variant specified.");
    }
}

template <typename precision, typename int_size>
AbstractClosestClusterUpdater<precision, int_size>*
  AbstractStrategyFactory<precision, int_size>::createClosestClusterUpdater(Variant variant)
{
    switch (variant)
    {
        case (SpecificCoreset):
            return new SerialClosestClusterUpdater<precision, int_size>(createClosestClusterFinder(Reg),
                                                                        createCoresetClusteringDataUpdater());
        default:
            return new SerialClosestClusterUpdater<precision, int_size>(createClosestClusterFinder(variant),
                                                                        createClusteringDataUpdater());
    }
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* AbstractStrategyFactory<precision, int_size>::createPointReassigner(
  Variant variant)
{
    switch (variant)
    {
        case (Reg):
            return createRegPointReassigner();
        case (Opt):
            return createOptPointReassigner();
        default:
            throw std::runtime_error("Invalid PointReassigner variant specified.");
    }
}

template <typename precision, typename int_size>
IMultiWeightedRandomSelector<precision, int_size>*
  AbstractStrategyFactory<precision, int_size>::createMultiWeightedRandomSelector()
{
    return new MultiWeightedRandomSelector<precision, int_size>();
}

template <typename precision, typename int_size>
IWeightedRandomSelector<precision, int_size>*
  AbstractStrategyFactory<precision, int_size>::createWeightedRandomSelector()
{
    return new SingleWeightedRandomSelector<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractClusteringDataUpdater<precision, int_size>*
  AbstractStrategyFactory<precision, int_size>::createCoresetClusteringDataUpdater()
{
    return new CoresetClusteringDataUpdater<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractClusteringDataUpdater<precision, int_size>*
  SerialStrategyFactory<precision, int_size>::createClusteringDataUpdater()
{
    return new ClusteringDataUpdater<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* SerialStrategyFactory<precision, int_size>::createRegPointReassigner()
{
    return new SerialPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* SerialStrategyFactory<precision, int_size>::createOptPointReassigner()
{
    return new SerialOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractWeightedAverager<precision, int_size>* SerialStrategyFactory<precision, int_size>::createWeightedAverager()
{
    return new SerialWeightedMultiVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractAverager<precision, int_size>* SerialStrategyFactory<precision, int_size>::createVectorAverager()
{
    return new SerialVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
IDistanceSumCalculator<precision, int_size>* SerialStrategyFactory<precision, int_size>::createDistanceSumCalculator()
{
    return new SerialDistanceSumCalculator<precision, int_size>();
}

template <typename precision, typename int_size>
ICoresetDistributionCalculator<precision, int_size>*
  SerialStrategyFactory<precision, int_size>::createCoresetDistributionCalculator()
{
    return new SerialCoresetDistributionCalculator<precision, int_size>();
}

template <typename precision, typename int_size>
IKmeansDataCreator<precision, int_size>* SerialStrategyFactory<precision, int_size>::createKmeansDataCreator()
{
    return new SharedMemoryKmeansDataCreator<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractCoresetClusteringFinisher<precision, int_size>*
  SerialStrategyFactory<precision, int_size>::createCoresetClusteringFinisher()
{
    return new SharedMemoryCoresetClusteringFinisher<precision, int_size>(
      this->createClosestClusterUpdater(SpecificCoreset));
}

template <typename precision, typename int_size>
AbstractClusteringDataUpdater<precision, int_size>*
  OMPStrategyFactory<precision, int_size>::createClusteringDataUpdater()
{
    return new AtomicClusteringDataUpdater<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* OMPStrategyFactory<precision, int_size>::createRegPointReassigner()
{
    return new OMPPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* OMPStrategyFactory<precision, int_size>::createOptPointReassigner()
{
    return new OMPOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractWeightedAverager<precision, int_size>* OMPStrategyFactory<precision, int_size>::createWeightedAverager()
{
    return new OMPWeightedMultiVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractAverager<precision, int_size>* OMPStrategyFactory<precision, int_size>::createVectorAverager()
{
    return new OMPVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
IDistanceSumCalculator<precision, int_size>* OMPStrategyFactory<precision, int_size>::createDistanceSumCalculator()
{
    return new OMPDistanceSumCalculator<precision, int_size>();
}

template <typename precision, typename int_size>
ICoresetDistributionCalculator<precision, int_size>*
  OMPStrategyFactory<precision, int_size>::createCoresetDistributionCalculator()
{
    return new OMPCoresetDistributionCalculator<precision, int_size>();
}

template <typename precision, typename int_size>
IKmeansDataCreator<precision, int_size>* OMPStrategyFactory<precision, int_size>::createKmeansDataCreator()
{
    return new SharedMemoryKmeansDataCreator<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractCoresetClusteringFinisher<precision, int_size>*
  OMPStrategyFactory<precision, int_size>::createCoresetClusteringFinisher()
{
    return new SharedMemoryCoresetClusteringFinisher<precision, int_size>(
      this->createClosestClusterUpdater(SpecificCoreset));
}

template <typename precision, typename int_size>
AbstractClusteringDataUpdater<precision, int_size>*
  MPIStrategyFactory<precision, int_size>::createClusteringDataUpdater()
{
    return new DistributedClusteringDataUpdater<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* MPIStrategyFactory<precision, int_size>::createRegPointReassigner()
{
    return new SerialPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* MPIStrategyFactory<precision, int_size>::createOptPointReassigner()
{
    return new SerialOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractWeightedAverager<precision, int_size>* MPIStrategyFactory<precision, int_size>::createWeightedAverager()

{
    return new SerialWeightedMultiVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractAverager<precision, int_size>* MPIStrategyFactory<precision, int_size>::createVectorAverager()
{
    return new SerialVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
IDistanceSumCalculator<precision, int_size>* MPIStrategyFactory<precision, int_size>::createDistanceSumCalculator()
{
    return new SerialDistanceSumCalculator<precision, int_size>();
}

template <typename precision, typename int_size>
ICoresetDistributionCalculator<precision, int_size>*
  MPIStrategyFactory<precision, int_size>::createCoresetDistributionCalculator()
{
    return nullptr;
}

template <typename precision, typename int_size>
IKmeansDataCreator<precision, int_size>* MPIStrategyFactory<precision, int_size>::createKmeansDataCreator()
{
    return new MPIKmeansDataCreator<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractCoresetClusteringFinisher<precision, int_size>*
  MPIStrategyFactory<precision, int_size>::createCoresetClusteringFinisher()
{
    return new MPICoresetClusteringFinisher<precision, int_size>(this->createClosestClusterUpdater(SpecificCoreset));
}

template <typename precision, typename int_size>
AbstractClusteringDataUpdater<precision, int_size>*
  HybridStrategyFactory<precision, int_size>::createClusteringDataUpdater()
{
    return new AtomicDistributedClusteringDataUpdater<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* HybridStrategyFactory<precision, int_size>::createRegPointReassigner()

{
    return new OMPPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractPointReassigner<precision, int_size>* HybridStrategyFactory<precision, int_size>::createOptPointReassigner()

{
    return new OMPOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
}

template <typename precision, typename int_size>
AbstractWeightedAverager<precision, int_size>* HybridStrategyFactory<precision, int_size>::createWeightedAverager()

{
    return new OMPWeightedMultiVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractAverager<precision, int_size>* HybridStrategyFactory<precision, int_size>::createVectorAverager()
{
    return new OMPVectorAverager<precision, int_size>();
}

template <typename precision, typename int_size>
IDistanceSumCalculator<precision, int_size>* HybridStrategyFactory<precision, int_size>::createDistanceSumCalculator()

{
    return new OMPDistanceSumCalculator<precision, int_size>();
}
template <typename precision, typename int_size>
ICoresetDistributionCalculator<precision, int_size>*
  HybridStrategyFactory<precision, int_size>::createCoresetDistributionCalculator()
{
    return nullptr;
}

template <typename precision, typename int_size>
IKmeansDataCreator<precision, int_size>* HybridStrategyFactory<precision, int_size>::createKmeansDataCreator()
{
    return new MPIKmeansDataCreator<precision, int_size>();
}

template <typename precision, typename int_size>
AbstractCoresetClusteringFinisher<precision, int_size>*
  HybridStrategyFactory<precision, int_size>::createCoresetClusteringFinisher()
{
    return new MPICoresetClusteringFinisher<precision, int_size>(this->createClosestClusterUpdater(SpecificCoreset));
}
}  // namespace HPKmeans