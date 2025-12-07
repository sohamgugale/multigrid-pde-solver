#include "common/matrix.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mg {

// ============================================================================
// Vector implementations
// ============================================================================

double Vector::norm2() const {
    double sum = 0.0;
    for (double val : data_) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

double Vector::norm_inf() const {
    double max_val = 0.0;
    for (double val : data_) {
        max_val = std::max(max_val, std::abs(val));
    }
    return max_val;
}

double Vector::dot(const Vector& other) const {
    if (size() != other.size()) {
        throw std::runtime_error("Vector size mismatch in dot product");
    }
    double sum = 0.0;
    for (size_t i = 0; i < size(); ++i) {
        sum += data_[i] * other.data_[i];
    }
    return sum;
}

void Vector::axpy(double a, const Vector& x) {
    if (size() != x.size()) {
        throw std::runtime_error("Vector size mismatch in axpy");
    }
    for (size_t i = 0; i < size(); ++i) {
        data_[i] += a * x.data_[i];
    }
}

void Vector::scale(double alpha) {
    for (double& val : data_) {
        val *= alpha;
    }
}

// ============================================================================
// Dense Matrix implementations
// ============================================================================

void Matrix::matvec(const Vector& x, Vector& y) const {
    if (x.size() != cols_ || y.size() != rows_) {
        throw std::runtime_error("Matrix-vector size mismatch");
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols_; ++j) {
            sum += (*this)(i, j) * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// Sparse Matrix implementations (CSR format)
// ============================================================================

void SparseMatrix::add_entry(size_t i, size_t j, double val) {
    if (finalized_) {
        throw std::runtime_error("Cannot add entries to finalized sparse matrix");
    }
    if (std::abs(val) > 1e-15) {  // Skip near-zero entries
        entries_.push_back({i, j, val});
    }
}

void SparseMatrix::finalize() {
    if (finalized_) return;
    
    // Sort entries by row, then column
    std::sort(entries_.begin(), entries_.end());
    
    // Build CSR structure
    row_ptr_.resize(rows_ + 1, 0);
    
    for (const auto& entry : entries_) {
        values_.push_back(entry.value);
        col_indices_.push_back(static_cast<int>(entry.col));
        row_ptr_[entry.row + 1]++;
    }
    
    // Convert counts to offsets
    for (size_t i = 0; i < rows_; ++i) {
        row_ptr_[i + 1] += row_ptr_[i];
    }
    
    // Clear temporary storage
    entries_.clear();
    entries_.shrink_to_fit();
    
    finalized_ = true;
}

void SparseMatrix::matvec(const Vector& x, Vector& y) const {
    if (!finalized_) {
        throw std::runtime_error("Must finalize sparse matrix before matvec");
    }
    if (x.size() != cols_ || y.size() != rows_) {
        throw std::runtime_error("Sparse matrix-vector size mismatch");
    }
    
    y.fill(0.0);
    
    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (int k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
            sum += values_[k] * x[col_indices_[k]];
        }
        y[i] = sum;
    }
}

} // namespace mg
