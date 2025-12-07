#include "cpu/smoothers.hpp"
#include <memory>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace mg {

std::unique_ptr<Smoother> Smoother::create(SmootherType type, double omega) {
    switch (type) {
        case SmootherType::JACOBI:
            return std::make_unique<JacobiSmoother>(omega);
        case SmootherType::GAUSS_SEIDEL:
            return std::make_unique<GaussSeidelSmoother>(omega);
        default:
            return std::make_unique<JacobiSmoother>(omega);
    }
}

// ============================================================================
// Jacobi Smoother: u^{k+1} = u^k + omega * D^{-1} * (b - A*u^k)
// ============================================================================

void JacobiSmoother::smooth(const SparseMatrix& A, Vector& u, 
                            const Vector& b, int nu) {
    size_t n = A.rows();
    Vector u_new(n);
    Vector residual(n);
    
    const auto& values = A.values();
    const auto& col_idx = A.col_indices();
    const auto& row_ptr = A.row_ptr();
    
    for (int iter = 0; iter < nu; ++iter) {
        // Compute residual and diagonal inverse
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < n; ++i) {
            double diag = 0.0;
            double Ax = 0.0;
            
            for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                int j = col_idx[k];
                if (i == static_cast<size_t>(j)) {
                    diag = values[k];
                }
                Ax += values[k] * u[j];
            }
            
            residual[i] = b[i] - Ax;
            
            if (std::abs(diag) > 1e-15) {
                u_new[i] = u[i] + omega_ * residual[i] / diag;
            } else {
                u_new[i] = u[i];
            }
        }
        
        // Update u
        u = u_new;
    }
}

// ============================================================================
// Gauss-Seidel Smoother: u_i^{k+1} = u_i^k + omega/a_ii * (b_i - (Au)_i)
// ============================================================================

void GaussSeidelSmoother::smooth(const SparseMatrix& A, Vector& u, 
                                 const Vector& b, int nu) {
    size_t n = A.rows();
    
    const auto& values = A.values();
    const auto& col_idx = A.col_indices();
    const auto& row_ptr = A.row_ptr();
    
    for (int iter = 0; iter < nu; ++iter) {
        // Forward sweep
        for (size_t i = 0; i < n; ++i) {
            double diag = 0.0;
            double Ax = 0.0;
            
            for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                int j = col_idx[k];
                if (i == static_cast<size_t>(j)) {
                    diag = values[k];
                }
                Ax += values[k] * u[j];
            }
            
            if (std::abs(diag) > 1e-15) {
                u[i] += omega_ * (b[i] - Ax) / diag;
            }
        }
    }
}

} // namespace mg
