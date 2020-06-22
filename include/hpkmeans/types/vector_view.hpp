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

    inline T& at(const int32_t idx) { return m_data.at(m_displacement + idx); }

    inline const T& at(const int32_t idx) const { return m_data.at(m_displacement + idx); }

    inline const int32_t viewSize() const { return m_length; }

    inline const int32_t size() const { return static_cast<int32_t>(m_data.size()); }

    inline T* const data() { return m_data.data(); }

    inline const T* const data() const { return m_data.data(); }

    inline iterator viewBegin() { return m_data.begin() + m_displacement; }

    inline iterator viewEnd() { return m_data.begin() + m_displacement + m_length; }

    inline iterator begin() { return m_data.begin(); }

    inline iterator end() { return m_data.end(); }

    inline const_iterator cviewBegin() const { return m_data.cbegin() + m_displacement; }

    inline const_iterator cviewEnd() const { return m_data.cbegin() + m_displacement + m_length; }

    inline const_iterator cbegin() const { return m_data.cbegin(); }

    inline const_iterator cend() const { return m_data.cend(); }

private:
    int32_t m_length;
    int32_t m_displacement;
    std::vector<T> m_data;
};
}  // namespace hpkmeans