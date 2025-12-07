#include "cpu/fd_discretizer.hpp"
#include "solvers/multigrid.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

void run_benchmark(int nx, int ny, int num_threads) {
    using namespace mg;
    
    #ifdef USE_OPENMP
    omp_set_num_threads(num_threads);
    #endif
    
    PDEParams params;
    FDDiscretizer2D disc(nx, ny, 0.0, 1.0, 0.0, 1.0, params, PDEType::HEAT);
    
    SparseMatrix A;
    Vector f;
    disc.assemble_poisson(A, f);
    
    MultigridParams mg_params;
    mg_params.max_levels = 1;
    mg_params.tolerance = 1e-6;
    mg_params.max_iterations = 100;
    mg_params.verbose = false;
    
    GeometricMultigrid mg(mg_params);
    mg.setup(A);
    
    Vector u(nx * ny);
    u.fill(0.0);
    
    Timer timer;
    mg.solve(A, u, f);
    double time = timer.elapsed();
    
    std::cout << nx << "," << ny << "," << num_threads << "," 
              << time << "," << mg.residual_history().size() - 1 << std::endl;
}

int main() {
    using namespace mg;
    
    Logger::instance().set_level(LogLevel::WARNING);
    
    std::cout << "nx,ny,threads,time,iterations" << std::endl;
    
    // Test with different thread counts
    for (int threads : {1, 2, 4}) {
        run_benchmark(65, 65, threads);
    }
    
    return 0;
}
