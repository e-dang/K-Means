#pragma once

#include <cstdint>
#include <vector>

namespace hpkmeans
{
template <typename T>
class VectorView
{
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef T value_type;

    VectorView() : m_length(0), m_displacement(0), m_data() {}

    VectorView(const int32_t totalSize, const int32_t length, const int32_t displacement, const T defaultVal = 0) :
        m_length(length), m_displacement(displacement), m_data(totalSize, defaultVal)
    {
    }

    VectorView(const VectorView& other) = default;

    VectorView(VectorView&& other) = default;

    VectorView& operator=(const VectorView& lhs) = default;

    VectorView& operator=(VectorView&& lhs) = default;

    T& operator[](const int32_t idx) { return m_data[m_displacement + idx]; }

    const T& operator[](const int32_t idx) const { return m_data[m_displacement + idx]; }

    T& at(const int32_t idx) { return m_data.at(m_displacement + idx); }

    const T& at(const int32_t idx) const { return m_data.at(m_displacement + idx); }

    const int32_t viewSize() const { return m_length; }

    const int32_t size() const { return static_cast<int32_t>(m_data.size()); }

    T* const data() { return m_data.data(); }

    const T* const data() const { return m_data.data(); }

    iterator viewBegin() { return m_data.begin() + m_displacement; }

    iterator viewEnd() { return m_data.begin() + m_displacement + m_length; }

    iterator begin() { return m_data.begin(); }

    iterator end() { return m_data.end(); }

    const_iterator cviewBegin() const { return m_data.cbegin() + m_displacement; }

    const_iterator cviewEnd() const { return m_data.cbegin() + m_displacement + m_length; }

    const_iterator cbegin() const { return m_data.cbegin(); }

    const_iterator cend() const { return m_data.cend(); }

private:
    int32_t m_length;
    int32_t m_displacement;
    std::vector<T> m_data;
};
}  // namespace hpkmeans