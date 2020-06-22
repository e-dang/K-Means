#pragma once

#include <matrix/matrix.hpp>
#include <vector>

namespace hpkmeans
{
template <typename T>
class Coreset
{
public:
    Coreset(const int32_t rows, const int32_t cols, const bool& autoReserve = false) :
        m_data(rows, cols, autoReserve), m_weights()
    {
        if (autoReserve)
            m_weights.resize(rows);
        else
            m_weights.reserve(rows);
    }

    Coreset(const Coreset& other) = delete;

    Coreset(Coreset&& other) = default;

    Coreset& operator=(const Coreset& lhs) = delete;

    Coreset& operator=(Coreset&& lhs) = default;

    template <typename Iter>
    inline void append(Iter begin, Iter end, const T& weight)
    {
        m_data.append(begin, end);
        m_weights.push_back(weight);
    }

    inline const int32_t numRows() const { return m_data.numRows(); }

    inline const int32_t cols() const { return m_data.cols(); }

    inline const int64_t size() const { return m_data.size(); }

    inline T* const data() { return m_data.data(); }

    inline const T* const data() const { return m_data.data(); }

    inline T* const weights() { return m_weights.data(); }

    inline const T* const weights() const { return m_weights.data(); }

    inline const Matrix<T>* const getData() const { return &m_data; }

    inline const std::vector<T>* const getWeights() const { return &m_weights; }

private:
    Matrix<T> m_data;
    std::vector<T> m_weights;
};
}  // namespace hpkmeans