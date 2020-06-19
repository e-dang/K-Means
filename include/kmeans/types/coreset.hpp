#pragma once

#include <matrix/matrix.hpp>
#include <vector>

namespace hpkmeans
{
template <typename T>
class Coreset
{
public:
    Coreset(const int32_t rows, const int32_t cols) : m_data(rows, cols), m_weights() { m_weights.reserve(rows); }

    Coreset(const Coreset& other) = delete;

    Coreset(Coreset&& other) = default;

    Coreset& operator=(const Coreset& lhs) = delete;

    Coreset& operator=(Coreset&& lhs) = default;

    template <typename Iter>
    void append(Iter begin, Iter end, const T& weight)
    {
        m_data.append(begin, end);
        m_weights.push_back(weight);
    }

    const Matrix<T>* const data() const { return &m_data; }

    const std::vector<T>* const weights() const { return &m_weights; }

private:
    Matrix<T> m_data;
    std::vector<T> m_weights;
};
}  // namespace hpkmeans