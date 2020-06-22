#pragma once

#include <hpkmeans/utils/uniform_selector.hpp>
#include <hpkmeans/utils/utils.hpp>
#include <set>

namespace hpkmeans
{
class UniformSelector
{
public:
    UniformSelector(const int64_t* seed = nullptr, const int min = 0) : m_seed(0), m_min(min)
    {
        if (seed == nullptr)
            m_seed = getTime();
        else
            m_seed = *seed;
    }

    int32_t selectSingle(const int32_t containerSize) const
    {
        auto selection = select(1, containerSize);
        return *selection.begin();
    }

    std::set<int32_t> select(const int32_t sampleSize, const int32_t containerSize) const
    {
        static RNGType rng(m_seed);
        static boost::random::uniform_int_distribution<> dist(m_min, containerSize - 1);
        static boost::variate_generator<RNGType, boost::random::uniform_int_distribution<>> gen(rng, dist);

        std::set<int32_t> selections;
        while (static_cast<int>(selections.size()) < sampleSize)
        {
            selections.insert(gen());
        }

        return selections;
    }

private:
    int64_t m_seed;
    int m_min;
};
}  // namespace hpkmeans