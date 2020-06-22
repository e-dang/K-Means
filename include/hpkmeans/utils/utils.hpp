#pragma once

#include <mpi.h>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <chrono>
#include <type_traits>

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

inline int getCommRank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

inline int getCommSize()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

template <typename T>
constexpr MPI_Datatype matchMPIType()
{
    static_assert(std::is_floating_point_v<T>, "Template parameter T must be a floating point type!");

    if constexpr (std::is_same_v<T, double>)
        return MPI_DOUBLE;

    return MPI_FLOAT;
}

template <typename Iter>
inline void print(Iter begin, Iter end)
{
    for (; begin != end; ++begin)
    {
        std::cout << *begin << " ";
    }
    std::cout << '\n' << std::flush;
}

constexpr bool isSharedMemory(Parallelism parallelism)
{
    return parallelism == Parallelism::Serial || parallelism == Parallelism::OMP;
}

constexpr bool isDistributed(Parallelism parallelism)
{
    return parallelism == Parallelism::MPI || parallelism == Parallelism::Hybrid;
}

constexpr bool isSingleThreaded(Parallelism parallelism)
{
    return parallelism == Parallelism::Serial || parallelism == Parallelism::MPI;
}

constexpr bool isMultiThreaded(Parallelism parallelism)
{
    return parallelism == Parallelism::OMP || parallelism == Parallelism::Hybrid;
}

template <Parallelism Level>
constexpr Parallelism getConjugateParallelism()
{
    if constexpr (Level == Parallelism::Serial)
        return Parallelism::MPI;
    else if constexpr (Level == Parallelism::OMP)
        return Parallelism::Hybrid;
    else if constexpr (Level == Parallelism::MPI)
        return Parallelism::Serial;
    else
        return Parallelism::OMP;
}

template <Parallelism Level, class Container>
inline std::enable_if_t<isSingleThreaded(Level), typename Container::value_type> accumulate(
  const Container* container, const typename Container::value_type& initVal = 0.0)
{
    return std::accumulate(container->cbegin(), container->cend(), initVal);
}

template <Parallelism Level, class Container>
inline std::enable_if_t<Level == Parallelism::OMP, typename Container::value_type> accumulate(
  const Container* container, const typename Container::value_type& initVal = 0.0)
{
    typename Container::value_type sum = 0.0;

#pragma omp parallel for schedule(static), reduction(+ : sum)
    for (int32_t i = 0; i < container->size(); ++i)
    {
        sum += container->at(i);
    }

    return sum;
}

template <Parallelism Level, class Container>
inline std::enable_if_t<Level == Parallelism::Hybrid, typename Container::value_type> accumulate(
  const Container* container, const typename Container::value_type& initVal = 0.0)
{
    typename Container::value_type sum = 0.0;

#pragma omp parallel for schedule(static), reduction(+ : sum)
    for (int32_t i = 0; i < container->viewSize(); ++i)
    {
        sum += container->at(i);
    }

    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, matchMPIType<typename Container::value_type>(), MPI_SUM, MPI_COMM_WORLD);
    return sum;
}
}  // namespace hpkmeans