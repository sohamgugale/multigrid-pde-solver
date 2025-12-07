#include "cpu/fd_discretizer.hpp"
#include <cmath>
#include <algorithm>

namespace mg {

FDDiscretizer2D::FDDiscretizer2D(int nx, int ny, 
                                 double xmin, double xmax,
                                 double ymin, double ymax, 
                                 const PDEParams& params,
                                 PDEType type)
    : nx_(nx), ny_(ny), 
      xmin_(xmin), xmax_(xmax), 
      ymin_(ymin), ymax_(ymax),
      params_(params), type_(type) {
    
    dx_ = (xmax_ - xmin_) / (nx_ - 1);
    dy_ = (ymax_ - ymin_) / (ny_ - 1);
}

void FDDiscretizer2D::assemble_poisson(SparseMatrix& A, Vector& f) {
    // Assemble -Laplacian(u) = f discretized with 5-point stencil
    // This is the core linear system for multigrid solver
    
    A.set_size(size(), size());
    f.resize(size());
    f.fill(1.0);  // Simple source term
    
    double dx2_inv = 1.0 / (dx_ * dx_);
    double dy2_inv = 1.0 / (dy_ * dy_);
    
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            int k = idx(i, j);
            
            // Boundary nodes - Dirichlet conditions
            if (i == 0 || i == nx_-1 || j == 0 || j == ny_-1) {
                A.add_entry(k, k, 1.0);
                f[k] = 0.0;  // Homogeneous BC
                continue;
            }
            
            // Interior: -d²u/dx² - d²u/dy² = f
            // 5-point stencil: [-1, -1, 4, -1, -1] / h²
            A.add_entry(k, k, 2.0 * (dx2_inv + dy2_inv));  // Center
            A.add_entry(k, idx(i-1, j), -dx2_inv);         // West
            A.add_entry(k, idx(i+1, j), -dx2_inv);         // East
            A.add_entry(k, idx(i, j-1), -dy2_inv);         // South
            A.add_entry(k, idx(i, j+1), -dy2_inv);         // North
        }
    }
    
    A.finalize();
}

void FDDiscretizer2D::assemble_system(SparseMatrix& A, Vector& b, double dt) {
    // Assemble system for time-dependent PDE
    // Backward Euler: (I - dt*L)u^{n+1} = u^n
    
    A.set_size(size(), size());
    b.resize(size());
    
    if (type_ == PDEType::HEAT) {
        // Heat equation: u_t = D * Laplacian(u)
        double D = params_.diffusion;
        double dx2_inv = 1.0 / (dx_ * dx_);
        double dy2_inv = 1.0 / (dy_ * dy_);
        double coef = D * dt;
        
        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                int k = idx(i, j);
                
                if (i == 0 || i == nx_-1 || j == 0 || j == ny_-1) {
                    A.add_entry(k, k, 1.0);
                    continue;
                }
                
                // (I - dt*D*Laplacian)
                double diag = 1.0 + coef * 2.0 * (dx2_inv + dy2_inv);
                A.add_entry(k, k, diag);
                A.add_entry(k, idx(i-1, j), -coef * dx2_inv);
                A.add_entry(k, idx(i+1, j), -coef * dx2_inv);
                A.add_entry(k, idx(i, j-1), -coef * dy2_inv);
                A.add_entry(k, idx(i, j+1), -coef * dy2_inv);
            }
        }
    }
    
    A.finalize();
}

void FDDiscretizer2D::set_initial_condition(Vector& u) {
    u.resize(size());
    
    if (type_ == PDEType::HEAT) {
        // Gaussian initial condition
        for (int i = 0; i < nx_; ++i) {
            double xi = x(i);
            for (int j = 0; j < ny_; ++j) {
                double yj = y(j);
                double r2 = (xi - 0.5) * (xi - 0.5) + (yj - 0.5) * (yj - 0.5);
                u[idx(i, j)] = std::exp(-50.0 * r2);
            }
        }
    } else {  // BLACK_SCHOLES
        // Option payoff: max(S - K, 0)
        for (int i = 0; i < nx_; ++i) {
            double xi = x(i);
            for (int j = 0; j < ny_; ++j) {
                int k = idx(i, j);
                double S = std::exp(xi);  // Log-price transform
                u[k] = std::max(S - params_.strike, 0.0);
            }
        }
    }
    
    set_boundary_conditions(u);
}

void FDDiscretizer2D::set_boundary_conditions(Vector& u) {
    // Homogeneous Dirichlet BC: u = 0 on boundary
    for (int i = 0; i < nx_; ++i) {
        u[idx(i, 0)] = 0.0;
        u[idx(i, ny_-1)] = 0.0;
    }
    for (int j = 0; j < ny_; ++j) {
        u[idx(0, j)] = 0.0;
        u[idx(nx_-1, j)] = 0.0;
    }
}

void FDDiscretizer2D::compute_exact_solution(Vector& u_exact, double t) {
    u_exact.resize(size());
    
    // For heat equation with Gaussian IC, we can compute semi-analytical solution
    // For now, just copy IC (can extend later)
    set_initial_condition(u_exact);
}

} // namespace mg
