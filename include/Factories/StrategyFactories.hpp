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

    IClosestClusterFinder<precision, int_size>* createClosestClusterFinder(Variant variant)
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

    AbstractClosestClusterUpdater<precision, int_size>* createClosestClusterUpdater(Variant variant)
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

    AbstractPointReassigner<precision, int_size>* createPointReassigner(Variant variant)
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

    IMultiWeightedRandomSelector<precision, int_size>* createMultiWeightedRandomSelector()
    {
        return new MultiWeightedRandomSelector<precision, int_size>();
    };

    IWeightedRandomSelector<precision, int_size>* createWeightedRandomSelector()
    {
        return new SingleWeightedRandomSelector<precision, int_size>();
    };

    AbstractClusteringDataUpdater<precision, int_size>* createCoresetClusteringDataUpdater()
    {
        return new CoresetClusteringDataUpdater<precision, int_size>();
    }

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
    SerialStrategyFactory() = default;

    ~SerialStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override
    {
        return new ClusteringDataUpdater<precision, int_size>();
    }

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override
    {
        return new SerialPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override
    {
        return new SerialOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override
    {
        return new SerialWeightedMultiVectorAverager<precision, int_size>();
    }

    AbstractAverager<precision, int_size>* createVectorAverager() override
    {
        return new SerialVectorAverager<precision, int_size>();
    }

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override
    {
        return new SerialDistanceSumCalculator<precision, int_size>();
    }

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override
    {
        return new SerialCoresetDistributionCalculator<precision, int_size>();
    }

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override
    {
        return new SharedMemoryKmeansDataCreator<precision, int_size>();
    }

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override
    {
        return new SharedMemoryCoresetClusteringFinisher<precision, int_size>(
          this->createClosestClusterUpdater(SpecificCoreset));
    }
};

template <typename precision = double, typename int_size = int32_t>
class OMPStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    OMPStrategyFactory() = default;

    ~OMPStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override
    {
        return new AtomicClusteringDataUpdater<precision, int_size>();
    }

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override
    {
        return new OMPPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override
    {
        return new OMPOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override
    {
        return new OMPWeightedMultiVectorAverager<precision, int_size>();
    }

    AbstractAverager<precision, int_size>* createVectorAverager() override
    {
        return new OMPVectorAverager<precision, int_size>();
    }

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override
    {
        return new OMPDistanceSumCalculator<precision, int_size>();
    }

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override
    {
        return new OMPCoresetDistributionCalculator<precision, int_size>();
    }

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override
    {
        return new SharedMemoryKmeansDataCreator<precision, int_size>();
    }

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override
    {
        return new SharedMemoryCoresetClusteringFinisher<precision, int_size>(
          this->createClosestClusterUpdater(SpecificCoreset));
    }
};

template <typename precision = double, typename int_size = int32_t>
class MPIStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    MPIStrategyFactory() = default;

    ~MPIStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override
    {
        return new DistributedClusteringDataUpdater<precision, int_size>();
    }

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override
    {
        return new SerialPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override
    {
        return new SerialOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override
    {
        return new SerialWeightedMultiVectorAverager<precision, int_size>();
    }

    AbstractAverager<precision, int_size>* createVectorAverager() override
    {
        return new SerialVectorAverager<precision, int_size>();
    }

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override
    {
        return new SerialDistanceSumCalculator<precision, int_size>();
    }

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override
    {
        return nullptr;
    }

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override
    {
        return new MPIKmeansDataCreator<precision, int_size>();
    }

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override
    {
        return new MPICoresetClusteringFinisher<precision, int_size>(
          this->createClosestClusterUpdater(SpecificCoreset));
    }
};

template <typename precision = double, typename int_size = int32_t>
class HybridStrategyFactory : public AbstractStrategyFactory<precision, int_size>
{
public:
    HybridStrategyFactory() = default;

    ~HybridStrategyFactory() = default;

    AbstractClusteringDataUpdater<precision, int_size>* createClusteringDataUpdater() override
    {
        return new AtomicDistributedClusteringDataUpdater<precision, int_size>();
    }

    AbstractPointReassigner<precision, int_size>* createRegPointReassigner() override
    {
        return new OMPPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractPointReassigner<precision, int_size>* createOptPointReassigner() override
    {
        return new OMPOptimizedPointReassigner<precision, int_size>(this->createClosestClusterUpdater(Reg));
    };

    AbstractWeightedAverager<precision, int_size>* createWeightedAverager() override
    {
        return new OMPWeightedMultiVectorAverager<precision, int_size>();
    }

    AbstractAverager<precision, int_size>* createVectorAverager() override
    {
        return new OMPVectorAverager<precision, int_size>();
    }

    IDistanceSumCalculator<precision, int_size>* createDistanceSumCalculator() override
    {
        return new OMPDistanceSumCalculator<precision, int_size>();
    }

    ICoresetDistributionCalculator<precision, int_size>* createCoresetDistributionCalculator() override
    {
        return nullptr;
    }

    IKmeansDataCreator<precision, int_size>* createKmeansDataCreator() override
    {
        return new MPIKmeansDataCreator<precision, int_size>();
    }

    AbstractCoresetClusteringFinisher<precision, int_size>* createCoresetClusteringFinisher() override
    {
        return new MPICoresetClusteringFinisher<precision, int_size>(
          this->createClosestClusterUpdater(SpecificCoreset));
    }
};
}  // namespace HPKmeans