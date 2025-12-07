#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace mg {

// Dense vector class
class Vector {
public:
    Vector(size_t n = 0) : data_(n, 0.0) {}
    
    double& operator[](size_t i) { return data_[i]; }
    const double& operator[](size_t i) const { return data_[i]; }
    
    size_t size() const { return data_.size(); }
    void resize(size_t n) { data_.resize(n, 0.0); }
    void fill(double val) { std::fill(data_.begin(), data_.end(), val); }
    
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
    
    // Vector operations
    double norm2() const;
    double norm_inf() const;
    double dot(const Vector& other) const;
    void axpy(double a, const Vector& x);  // this = this + a*x
    void scale(double alpha);
    
private:
    std::vector<double> data_;
};

// Dense matrix class (for small systems)
class Matrix {
public:
    Matrix(size_t rows = 0, size_t cols = 0) 
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
    
    double& operator()(size_t i, size_t j) { 
        return data_[i * cols_ + j]; 
    }
    
    const double& operator()(size_t i, size_t j) const { 
        return data_[i * cols_ + j]; 
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    void resize(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols, 0.0);
    }
    
    void matvec(const Vector& x, Vector& y) const;
    
private:
    size_t rows_, cols_;
    std::vector<double> data_;
};

// Sparse matrix in CSR (Compressed Sparse Row) format
class SparseMatrix {
public:
    SparseMatrix() : rows_(0), cols_(0), finalized_(false) {}
    
    void set_size(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
    }
    
    void add_entry(size_t i, size_t j, double val);
    void finalize();
    void matvec(const Vector& x, Vector& y) const;
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t nnz() const { return values_.size(); }
    
    // Access to raw data (for advanced usage)
    const std::vector<double>& values() const { return values_; }
    const std::vector<int>& col_indices() const { return col_indices_; }
    const std::vector<int>& row_ptr() const { return row_ptr_; }
    
private:
    size_t rows_, cols_;
    bool finalized_;
    
    // CSR storage
    std::vector<double> values_;
    std::vector<int> col_indices_;
    std::vector<int> row_ptr_;
    
    // Temporary storage before finalization (COO format)
    struct Entry {
        size_t row, col;
        double value;
        bool operator<(const Entry& other) const {
            if (row != other.row) return row < other.row;
            return col < other.col;
        }
    };
    std::vector<Entry> entries_;
};

} // namespace mg
