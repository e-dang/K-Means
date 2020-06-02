#pragma once

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <chrono>

typedef boost::mt19937 RNGType;

namespace hpkmeans
{
inline int64_t getTime()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline double getRandFraction()
{
    thread_local static RNGType rng(getTime());
    thread_local static boost::uniform_real<double> distr(0, 1);
    thread_local static boost::variate_generator<RNGType, boost::uniform_real<double>> gen(rng, distr);

    return gen();
}
}  // namespace hpkmeans