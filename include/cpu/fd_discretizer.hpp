#pragma once
#include "common/matrix.hpp"
#include <array>
#include <string>

namespace mg {

struct PDEParams {
    double volatility = 0.2;       // σ (Black-Scholes volatility)
    double risk_free_rate = 0.05;  // r (risk-free interest rate)
    double strike = 100.0;         // K (strike price)
    double maturity = 1.0;         // T (time to maturity)
    
    // For heat equation: diffusion coefficient
    double diffusion = 1.0;
};

enum class PDEType {
    HEAT,           // Heat equation: u_t = D * Laplacian(u)
    BLACK_SCHOLES   // Black-Scholes: V_t + 0.5*σ²*S²*V_SS + r*S*V_S - r*V = 0
};

class FDDiscretizer2D {
public:
    FDDiscretizer2D(int nx, int ny, 
                    double xmin, double xmax, 
                    double ymin, double ymax, 
                    const PDEParams& params,
                    PDEType type = PDEType::HEAT);
    
    // Build system matrix for implicit time stepping
    void assemble_system(SparseMatrix& A, Vector& b, double dt);
    
    // Build right-hand side for Poisson equation: -Laplacian(u) = f
    void assemble_poisson(SparseMatrix& A, Vector& f);
    
    // Set initial conditions
    void set_initial_condition(Vector& u);
    
    // Apply boundary conditions
    void set_boundary_conditions(Vector& u);
    
    // Compute exact solution (for validation)
    void compute_exact_solution(Vector& u_exact, double t = 0.0);
    
    // Grid accessors
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int size() const { return nx_ * ny_; }
    
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    
    double xmin() const { return xmin_; }
    double xmax() const { return xmax_; }
    double ymin() const { return ymin_; }
    double ymax() const { return ymax_; }
    
    // Grid point coordinates
    double x(int i) const { return xmin_ + i * dx_; }
    double y(int j) const { return ymin_ + j * dy_; }
    
    // Linear index
    int idx(int i, int j) const { return i * ny_ + j; }
    
private:
    int nx_, ny_;
    double xmin_, xmax_, ymin_, ymax_;
    double dx_, dy_;
    PDEParams params_;
    PDEType type_;
};

} // namespace mg
