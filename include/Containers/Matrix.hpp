#pragma once

#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "Containers/Definitions.hpp"

// /**
//  * @brief Wrapper class around std::vector<value_t> that helps manipulate the vector as if it were a 2D nested
//  vector.
//  *        Keeping the data as a 1D vector rather than a nested vector makes it easier to implement with MPI.
//  */
// class Matrix
// {
// private:
//     // Member variables
//     std::vector<value_t> mData;
//     int32_t mNumRows;  // this is the number of datapoints in the matrix
//     int32_t mNumCols;  // this is the number of features of each datapoint in the matrix

// public:
//     Matrix() {}

//     Matrix(const int32_t& size, const int32_t& numRows, const int32_t& numCols)
//     {
//         mData.resize(size);
//         mNumRows = numRows;
//         mNumCols = numCols;
//     }

//     Matrix(const int32_t& numRows, const int32_t& numCols)
//     {
//         mData.reserve(numRows * numCols);
//         mNumRows = numRows;
//         mNumCols = numCols;
//     }

//     Matrix(std::vector<value_t> data, const int32_t& numRows, const int32_t& numCols)
//     {
//         if (static_cast<int32_t>(data.size()) > numRows * numCols)
//         {
//             throw std::runtime_error("The vector has more data than was specified.");
//         }

//         mData = data;
//         mData.reserve(numRows * numCols);
//         mNumRows = numRows;
//         mNumCols = numCols;
//     }

//     /**
//      * @brief Destroy the Matrix object
//      */
//     ~Matrix(){};

//     /**
//      * @brief Function to access the beginning of a "row" of the matrix, where each row of the matrix is a datapoint.
//      *        This function returns an iterator to the beginning of the row which can be used to access the rest of
//      the
//      *        row.
//      *
//      * @param row - The number of the row to retrieve.
//      * @return value_t *
//      */
//     value_t* at(const int32_t& row) { return mData.data() + (row * mNumCols); }

//     /**
//      * @brief Function to access a specific value in the matrix.
//      *
//      * @param row - The row number that the value is in.
//      * @param col - The column number that the value is in.
//      * @return value_t&
//      */
//     value_t& at(const int32_t& row, const int32_t& col) { return *(this->at(row) + col); }

//     const value_t at(const int32_t& row, const int32_t& col) const { return *(this->at(row) + col); }

//     const value_t* at(const int32_t& row) const { return mData.data() + (row * mNumCols); }

//     std::vector<value_t>::iterator begin() { return mData.begin(); }
//     std::vector<value_t>::iterator end() { return mData.end(); }

//     void appendDataPoint(const value_t* datapoint)
//     {
//         if (getNumData() < mNumRows)
//         {
//             std::copy(datapoint, datapoint + mNumCols, std::back_inserter(mData));
//         }
//         else
//         {
//             throw std::overflow_error("Cannot append to full matrix");
//         }
//     }

//     void fill(const int32_t& val) { std::fill(mData.begin(), mData.end(), val); }

//     void resize(const int32_t& val)
//     {
//         mData.resize(val * mNumCols);
//         mNumRows = val;
//     }

//     void reserve(const int32_t& val)
//     {
//         mData.resize(val * mNumCols);
//         if (val > mNumRows)
//             mNumRows = val;
//     }

//     value_t* data() { return mData.data(); }

//     int32_t size() const { return mData.size(); }
//     int32_t getNumData() const { return mData.size() / mNumCols; }
//     int32_t getMaxNumData() const { return mNumRows; }
//     int32_t getNumFeatures() const { return mNumCols; }

//     void operator=(const Matrix& lhs)
//     {
//         mData    = std::move(lhs.mData);
//         mNumRows = lhs.mNumRows;
//         mNumCols = lhs.mNumCols;
//     }

//     void display()
//     {
//         for (const auto& val : mData)
//         {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     }
// };
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class Matrix
{
private:
    precision* p_Data;
    int_size m_Rows;  // this is the number of datapoints in the matrix
    int_size m_Cols;  // this is the number of features of each datapoint in the matrix
    int_size m_Capacity;
    int_size m_Size;

public:
    class iterator
    {
    private:
        precision* m_ptr;

    public:
        iterator(precision* ptr) noexcept : m_ptr(ptr) {}

        ~iterator() = default;

        iterator& operator++()
        {
            m_ptr++;
            return *this;
        }

        iterator operator++(int)
        {
            iterator temp = *this;
            m_ptr++;
            return temp;
        }

        iterator& operator--()
        {
            m_ptr--;
            return *this;
        }

        iterator operator--(int)
        {
            auto temp = *this;
            m_ptr--;
            return temp;
        }

        precision& operator*() { return *m_ptr; }

        precision* operator->() { return m_ptr; }

        bool operator==(const iterator& rhs) { return m_ptr == rhs.m_ptr; }

        bool operator!=(const iterator& rhs) { return m_ptr != rhs.m_ptr; }
    };

    class const_iterator
    {
    private:
        precision* m_ptr;

    public:
        const_iterator(precision* ptr) noexcept : m_ptr(ptr) {}

        ~const_iterator() = default;

        const_iterator operator++()
        {
            m_ptr++;
            return *this;
        }

        const_iterator operator++(int)
        {
            auto temp = *this;
            m_ptr++;
            return temp;
        }

        const_iterator& operator--()
        {
            m_ptr--;
            return *this;
        }

        const_iterator operator--(int)
        {
            auto temp = *this;
            m_ptr--;
            return temp;
        }

        const precision& operator*() { return *m_ptr; }

        const precision* operator->() { return m_ptr; }

        bool operator==(const const_iterator& rhs) { return m_ptr == rhs.m_ptr; }

        bool operator!=(const const_iterator& rhs) { return m_ptr != rhs.m_ptr; }
    };

    Matrix() noexcept : p_Data(nullptr), m_Rows(0), m_Cols(0), m_Capacity(0), m_Size(0) {}

    Matrix(const int_size& numRows, const int_size& numCols, bool autoReserve = true, const int_size initVal = 0.0) :
        p_Data(nullptr), m_Rows(numRows), m_Cols(numCols), m_Capacity(numRows * numCols), m_Size(0)
    {
        if (!isValidNumRows() || !isValidNumCols() || !isValidNumCols())
        {
            throw std::invalid_argument("Invalid arguments passed to Matrix constructor! The number of rows ( " +
                                        std::to_string(numRows) + ") and number of columns " + std::to_string(numCols) +
                                        " must both be positive.");
        }

        p_Data = new precision[m_Capacity];
        if (autoReserve)
        {
            reserve(m_Rows);
            fill(initVal);
        }
    }

    Matrix(const Matrix& other) noexcept :
        p_Data(new precision[other.m_Capacity]),
        m_Rows(other.m_Rows),
        m_Cols(other.m_Cols),
        m_Capacity(other.m_Capacity),
        m_Size(other.m_Size)
    {
        std::copy(other.p_Data, other.p_Data + other.elements(), p_Data);
    }

    Matrix(Matrix&& other) noexcept : p_Data(nullptr), m_Rows(0), m_Cols(0), m_Capacity(0), m_Size(0)
    {
        *this = std::move(other);
    }

    ~Matrix()
    {
        delete[] p_Data;
        p_Data = nullptr;
    }

    Matrix& operator=(const Matrix& rhs)
    {
        if (this != &rhs)
        {
            delete[] p_Data;

            p_Data = new precision[rhs.m_Size];
            std::copy(rhs.p_Data, rhs.p_Data + rhs.elements(), p_Data);
            m_Rows     = rhs.m_Rows;
            m_Cols     = rhs.m_Cols;
            m_Capacity = rhs.m_Capacity;
            m_Size     = rhs.m_Size;
        }

        return *this;
    }

    Matrix& operator=(Matrix&& rhs)
    {
        if (this != &rhs)
        {
            delete[] p_Data;

            p_Data     = rhs.p_Data;
            m_Rows     = rhs.m_Rows;
            m_Cols     = rhs.m_Cols;
            m_Capacity = rhs.m_Capacity;
            m_Size     = rhs.m_Size;

            rhs.p_Data     = nullptr;
            rhs.m_Rows     = 0;
            rhs.m_Cols     = 0;
            rhs.m_Capacity = 0;
            rhs.m_Size     = 0;
        }

        return *this;
    }

    inline precision* data() noexcept { return p_Data; }
    inline const precision* data() const noexcept { return p_Data; }

    inline const int_size capacity() const noexcept { return m_Capacity; }
    inline const int_size rows() const noexcept { return m_Rows; }
    inline const int_size cols() const noexcept { return m_Cols; }
    inline int_size elements() const noexcept { return m_Size * m_Cols; }
    inline int_size size() const noexcept { return m_Size; }
    inline bool empty() const noexcept { return m_Size; }
    inline void reserve(const int_size& space)
    {
        if (space < 0 || (m_Size + space) * m_Cols > m_Capacity)
            throw std::length_error("Cannot reserve space less than zero or greater than the capacity.");

        m_Size += space;
    }

    inline precision* at(const int_size& row)
    {
#ifdef DEBUG_
        if (row >= m_Rows || row < 0)
            throw std::out_of_range("Row index " + std::to_string(row) + " out of range (max row index is " +
                                    std::to_string(m_Rows) + ").");
#endif
        return p_Data + (row * m_Cols);
    }

    inline const precision* at(const int_size& row) const
    {
#ifdef DEBUG_
        if (row >= m_Rows || row < 0)
            throw std::out_of_range("Row index " + std::to_string(row) + " out of range (max row index is " +
                                    std::to_string(m_Rows) + ").");
#endif
        return p_Data + (row * m_Cols);
    }

    inline precision& at(const int_size& row, const int_size& col)
    {
#ifdef DEBUG_
        if (col >= mCols || col < 0)
        {
            throw std::out_of_range("Column index " + std::to_string(col) + " out of range (max column index is " +
                                    std::to_string(mCols) + ").");
        }
#endif

        if (row + 1 > m_Size && row < m_Rows)
        {
            m_Size = row + 1;
        }

        return *(this->at(row) + col);
    }

    inline const precision& at(const int_size& row, const int_size& col) const
    {
#ifdef DEBUG_
        if (col >= m_Cols || col < 0)
        {
            throw std::out_of_range("Column index " + std::to_string(col) + " out of range (max column index is " +
                                    std::to_string(m_Cols) + ").");
        }
#endif

        return *(this->at(row) + col);
    }

    inline iterator begin() { return iterator(p_Data); }
    inline iterator end() { return iterator(p_Data + elements()); }
    inline iterator begin() const { return iterator(p_Data); }
    inline iterator end() const { return iterator(p_Data + elements()); }
    inline const_iterator cbegin() const { return const_iterator(p_Data); }
    inline const_iterator cend() const { return const_iterator(p_Data + elements()); }

    void push_back(const precision* datapoint)
    {
        if (elements() >= m_Capacity)
        {
            throw std::length_error("Cannot add data to full Matrix.");
        }

        std::copy(datapoint, datapoint + m_Cols, p_Data + elements());
        m_Size++;
    }

    inline void fill(const int_size& val) { std::fill(begin(), end(), val); }

    inline friend std::ostream& operator<<(std::ostream& os, const Matrix& lhs)
    {
        for (const auto& val : lhs)
        {
            os << val << " ";
        }
        return os;
    }

private:
    inline bool isValidNumRows()
    {
        if (m_Rows < 0)
            return false;

        return true;
    }

    inline bool isValidNumCols()
    {
        if (m_Cols < 0)
            return false;

        return true;
    }

    inline bool isValidCapacity()
    {
        if (m_Capacity < 0)
            return false;
        return true;
    }
};
}  // namespace HPKmeans
