#include "solvers/multigrid.hpp"
#include "utils/logger.hpp"
#include <cmath>
#include <iostream>

namespace mg {

GeometricMultigrid::GeometricMultigrid(const MultigridParams& params)
    : params_(params), num_levels_(0) {
    smoother_ = Smoother::create(params_.smoother, params_.smoother_omega);
}

void GeometricMultigrid::setup(const SparseMatrix& A_fine) {
    // For now, just use the fine grid - single level multigrid = preconditioned Richardson
    num_levels_ = 1;
    
    A_hierarchy_.clear();
    A_hierarchy_.push_back(A_fine);
    
    grid_sizes_.clear();
    grid_sizes_.push_back(A_fine.rows());
    
    rhs_.resize(num_levels_);
    solution_.resize(num_levels_);
    residual_.resize(num_levels_);
    
    for (int level = 0; level < num_levels_; ++level) {
        int n = grid_sizes_[level];
        rhs_[level].resize(n);
        solution_[level].resize(n);
        residual_[level].resize(n);
    }
    
    Logger::instance().info("Multigrid setup: ", num_levels_, " levels (single-level smoother)");
    Logger::instance().info("  Level 0: ", grid_sizes_[0], " unknowns");
}

void GeometricMultigrid::solve(const SparseMatrix& A, Vector& u, const Vector& b) {
    res_history_.clear();
    
    Vector r(b.size());
    A.matvec(u, r);
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = b[i] - r[i];
    }
    double res0 = r.norm2();
    double res = res0;
    
    res_history_.push_back(res);
    
    if (params_.verbose) {
        std::cout << "Initial residual: " << res0 << std::endl;
    }
    
    // Use stationary iteration (smoother) instead of multigrid for now
    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        // Apply smoother
        smoother_->smooth(A, u, b, params_.nu_pre + params_.nu_post);
        
        // Compute residual
        A.matvec(u, r);
        for (size_t i = 0; i < r.size(); ++i) {
            r[i] = b[i] - r[i];
        }
        res = r.norm2();
        res_history_.push_back(res);
        
        if (params_.verbose && (iter < 5 || iter % 5 == 0)) {
            std::cout << "Iteration " << iter + 1 << ": residual = " << res 
                      << " (reduction: " << res/res0 << ")" << std::endl;
        }
        
        if (res < params_.tolerance * res0 || res < params_.tolerance) {
            if (params_.verbose) {
                std::cout << "Converged in " << iter + 1 << " iterations" << std::endl;
                std::cout << "Final residual: " << res << std::endl;
            }
            break;
        }
        
        if (res > 100 * res0 || std::isnan(res)) {
            if (params_.verbose) {
                std::cout << "WARNING: Slow convergence or divergence detected" << std::endl;
            }
            break;
        }
    }
}

void GeometricMultigrid::cycle(int level, Vector& u, const Vector& b) {
    // Not used in single-level version
    smoother_->smooth(A_hierarchy_[level], u, b, params_.nu_pre + params_.nu_post);
}

void GeometricMultigrid::restrict(const Vector& fine, Vector& coarse, int level) {
    // Placeholder
}

void GeometricMultigrid::prolongate(const Vector& coarse, Vector& fine, int level) {
    // Placeholder
}

SparseMatrix GeometricMultigrid::galerkin_coarsening(const SparseMatrix& A_fine, int level) {
    return A_fine; // Placeholder
}

void GeometricMultigrid::solve_coarse(Vector& u, const Vector& b) {
    smoother_->smooth(A_hierarchy_.back(), u, b, 100);
}

} // namespace mg
