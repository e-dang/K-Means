#pragma once

#include <cmath>
#include <iostream>
#include <numeric>

template <typename T>
class L2Norm
{
public:
    static L2Norm& instance()
    {
        if (sp_instance == nullptr)
            sp_instance = new L2Norm<T>();

        return *sp_instance;
    }

    template <typename Iter1, typename Iter2>
    T operator()(const Iter1 p1Begin, const Iter1 p1End, const Iter2 p2Begin, const Iter2) const
    {
        auto result = std::inner_product(p1Begin, p1End, p2Begin, 0.0, std::plus<>(),
                                         [](const T val1, const T val2) { return std::pow(val1 - val2, 2); });

        return std::sqrt(result);
    }

private:
    static L2Norm<T>* sp_instance;
};

template <typename T>
L2Norm<T>* L2Norm<T>::sp_instance = nullptr;