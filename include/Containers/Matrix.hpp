#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include "Containers/Definitions.hpp"

/**
 * @brief Wrapper class around std::vector<value_t> that helps manipulate the vector as if it were a 2D nested vector.
 *        Keeping the data as a 1D vector rather than a nested vector makes it easier to implement with MPI.
 */
class Matrix
{
private:
    // Member variables
    std::vector<value_t> mData;
    int_fast32_t mNumRows;  // this is the number of datapoints in the matrix
    int_fast32_t mNumCols;  // this is the number of features of each datapoint in the matrix

public:
    Matrix() {}

    Matrix(const int_fast32_t& size, const int_fast32_t& numRows, const int_fast32_t& numCols)
    {
        mData.resize(size);
        mNumRows = numRows;
        mNumCols = numCols;
    }

    Matrix(const int_fast32_t& numRows, const int_fast32_t& numCols)
    {
        mData.reserve(numRows * numCols);
        mNumRows = numRows;
        mNumCols = numCols;
    }

    Matrix(std::vector<value_t> data, const int_fast32_t& numRows, const int_fast32_t& numCols)
    {
        if (static_cast<int_fast32_t>(data.size()) > numRows * numCols)
        {
            throw std::runtime_error("The vector has more data than was specified.");
        }

        mData = data;
        mData.reserve(numRows * numCols);
        mNumRows = numRows;
        mNumCols = numCols;
    }

    /**
     * @brief Destroy the Matrix object
     */
    ~Matrix(){};

    /**
     * @brief Function to access the beginning of a "row" of the matrix, where each row of the matrix is a datapoint.
     *        This function returns an iterator to the beginning of the row which can be used to access the rest of the
     *        row.
     *
     * @param row - The number of the row to retrieve.
     * @return value_t *
     */
    value_t* at(const int_fast32_t& row) { return mData.data() + (row * mNumCols); }

    /**
     * @brief Function to access a specific value in the matrix.
     *
     * @param row - The row number that the value is in.
     * @param col - The column number that the value is in.
     * @return value_t&
     */
    value_t& at(const int_fast32_t& row, const int_fast32_t& col) { return *(this->at(row) + col); }

    const value_t at(const int_fast32_t& row, const int_fast32_t& col) const { return *(this->at(row) + col); }

    const value_t* at(const int_fast32_t& row) const { return mData.data() + (row * mNumCols); }

    std::vector<value_t>::iterator begin() { return mData.begin(); }
    std::vector<value_t>::iterator end() { return mData.end(); }

    void appendDataPoint(const value_t* datapoint)
    {
        if (getNumData() < mNumRows)
        {
            std::copy(datapoint, datapoint + mNumCols, std::back_inserter(mData));
        }
        else
        {
            throw std::overflow_error("Cannot append to full matrix");
        }
    }

    void fill(const int_fast32_t& val) { std::fill(mData.begin(), mData.end(), val); }

    void resize(const int_fast32_t& val)
    {
        mData.resize(val * mNumCols);
        mNumRows = val;
    }

    void reserve(const int_fast32_t& val)
    {
        mData.resize(val * mNumCols);
        if (val > mNumRows)
            mNumRows = val;
    }

    value_t* data() { return mData.data(); }

    int_fast32_t size() const { return mData.size(); }
    int_fast32_t getNumData() const { return mData.size() / mNumCols; }
    int_fast32_t getMaxNumData() const { return mNumRows; }
    int_fast32_t getNumFeatures() const { return mNumCols; }

    void operator=(const Matrix& lhs)
    {
        mData    = std::move(lhs.mData);
        mNumRows = lhs.mNumRows;
        mNumCols = lhs.mNumCols;
    }

    void display()
    {
        for (const auto& val : mData)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};