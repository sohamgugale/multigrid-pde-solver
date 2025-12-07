#include "cpu/fd_discretizer.hpp"
#include "solvers/multigrid.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace mg;
    
    Logger::instance().set_level(LogLevel::INFO);
    Logger::instance().info("=== 2D Multigrid Poisson Solver Test ===\n");
    
    int nx = 65, ny = 65;
    Logger::instance().info("Grid size: ", nx, " x ", ny, " = ", nx*ny, " unknowns");
    
    PDEParams params;
    FDDiscretizer2D disc(nx, ny, 0.0, 1.0, 0.0, 1.0, params, PDEType::HEAT);
    
    Timer timer;
    SparseMatrix A;
    Vector f;
    
    Logger::instance().info("\nAssembling system...");
    disc.assemble_poisson(A, f);
    Logger::instance().info("Assembly time: ", timer.elapsed(), " seconds");
    Logger::instance().info("Matrix size: ", A.rows(), " x ", A.rows());
    Logger::instance().info("Non-zeros: ", A.nnz());
    
    MultigridParams mg_params;
    mg_params.max_levels = 6;
    mg_params.nu_pre = 2;
    mg_params.nu_post = 2;
    mg_params.smoother = SmootherType::GAUSS_SEIDEL;
    mg_params.cycle = CycleType::V_CYCLE;
    mg_params.tolerance = 1e-8;
    mg_params.max_iterations = 50;
    mg_params.verbose = true;
    
    GeometricMultigrid mg(mg_params);
    
    Logger::instance().info("\nSetting up multigrid hierarchy...");
    timer.reset();
    mg.setup(A);
    Logger::instance().info("Setup time: ", timer.elapsed(), " seconds\n");
    
    Vector u(nx * ny);
    u.fill(0.0);
    
    Logger::instance().info("Solving system...\n");
    timer.reset();
    mg.solve(A, u, f);
    double solve_time = timer.elapsed();
    
    Logger::instance().info("\n=== Results ===");
    Logger::instance().info("Total solve time: ", solve_time, " seconds");
    Logger::instance().info("Iterations: ", mg.residual_history().size() - 1);
    Logger::instance().info("Final residual: ", mg.residual_history().back());
    
    Vector r(f.size());
    A.matvec(u, r);
    for (size_t i = 0; i < r.size(); ++i) {
        r[i] = f[i] - r[i];
    }
    double res_norm = r.norm2();
    double rhs_norm = f.norm2();
    
    Logger::instance().info("||r||_2: ", res_norm);
    Logger::instance().info("||r||_2 / ||b||_2: ", res_norm / rhs_norm);
    
    std::cout << "\nConvergence history:" << std::endl;
    const auto& history = mg.residual_history();
    for (size_t i = 0; i < history.size(); ++i) {
        std::cout << "  Iteration " << std::setw(3) << i << ": "
                  << std::scientific << std::setprecision(6) 
                  << history[i] << std::endl;
    }
    
    Logger::instance().info("\n=== Test PASSED ===");
    
    return 0;
}
