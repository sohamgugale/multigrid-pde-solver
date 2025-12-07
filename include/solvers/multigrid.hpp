#pragma once
#include "common/matrix.hpp"
#include "cpu/smoothers.hpp"
#include <vector>
#include <memory>

namespace mg {

enum class CycleType {
    V_CYCLE,
    W_CYCLE
};

struct MultigridParams {
    int max_levels = 10;
    int nu_pre = 2;          // Pre-smoothing iterations
    int nu_post = 2;         // Post-smoothing iterations
    int nu_coarse = 10;      // Coarse grid solver iterations
    SmootherType smoother = SmootherType::GAUSS_SEIDEL;
    double smoother_omega = 1.0;
    CycleType cycle = CycleType::V_CYCLE;
    double tolerance = 1e-6;
    int max_iterations = 100;
    bool verbose = true;
};

class GeometricMultigrid {
public:
    GeometricMultigrid(const MultigridParams& params = MultigridParams());
    
    // Setup multigrid hierarchy
    void setup(const SparseMatrix& A_fine);
    
    // Solve Au = b using multigrid
    void solve(const SparseMatrix& A, Vector& u, const Vector& b);
    
    // Single V-cycle or W-cycle
    void cycle(int level, Vector& u, const Vector& b);
    
    // Get convergence history
    const std::vector<double>& residual_history() const { return res_history_; }
    
private:
    // Grid transfer operators
    void restrict(const Vector& fine, Vector& coarse, int level);
    void prolongate(const Vector& coarse, Vector& fine, int level);
    
    // Build coarse grid operators
    SparseMatrix galerkin_coarsening(const SparseMatrix& A_fine, int level);
    
    // Direct solver for coarsest level
    void solve_coarse(Vector& u, const Vector& b);
    
    MultigridParams params_;
    
    // Hierarchy data structures
    std::vector<SparseMatrix> A_hierarchy_;
    std::vector<Vector> rhs_;
    std::vector<Vector> solution_;
    std::vector<Vector> residual_;
    
    std::vector<int> grid_sizes_;
    
    std::unique_ptr<Smoother> smoother_;
    
    std::vector<double> res_history_;
    int num_levels_;
};

} // namespace mg
