cat > README.md << 'EOF'
# Parallel Multigrid/CUDA Solver for High-Dimensional Option Pricing PDEs

A high-performance numerical PDE solver implementing geometric multigrid methods with CPU (MPI/OpenMP) and GPU (CUDA) acceleration for pricing multi-dimensional financial derivatives.

## Features

- ✨ **Multigrid Solvers**: Geometric multigrid (GMG) with V-cycle and W-cycle
- 🚀 **Parallel Computing**: MPI domain decomposition, OpenMP threading, CUDA GPU acceleration
- 📊 **PDE Support**: 2D-4D Black-Scholes equations, heat equation
- 📈 **Performance Analysis**: Weak/strong scaling, GPU vs CPU comparisons
- 🧪 **Validation**: Unit tests, convergence tests, analytical comparisons

## Quick Start
```bash
# Clone the repository
git clone git@github.com:yourusername/multigrid-pde-solver.git
cd multigrid-pde-solver

# Build CPU version with OpenMP
mkdir build && cd build
cmake .. -DUSE_OPENMP=ON -DCMAKE_CXX_COMPILER=g++-13
make -j4

# Run 2D test
./bin/test_mg_2d

# Run benchmarks
./bin/benchmark_scaling --threads 4
```

## Requirements

- macOS 11.0+ (for development) or Linux (for CUDA)
- CMake 3.18+
- C++17 compiler (GCC 9+ recommended)
- OpenMPI 4.0+ (for MPI builds)
- Python 3.10+ (for analysis)
- CUDA 11.0+ (optional, Linux only)

## Build Options
```bash
cmake .. -DUSE_OPENMP=ON    # Enable OpenMP (default: ON)
cmake .. -DUSE_MPI=ON       # Enable MPI (default: OFF)
cmake .. -DUSE_CUDA=ON      # Enable CUDA (default: OFF, requires Linux/NVIDIA GPU)
```

## Project Structure
multigrid-pde-solver/
├── src/              # Source files
├── include/          # Header files
├── tests/            # Unit tests
├── benchmarks/       # Performance benchmarks
├── scripts/          # Python analysis scripts
├── results/          # Output plots and logs
└── docs/             # Documentation

## Documentation

- [Implementation Guide](docs/implementation.md)
- [Performance Results](docs/performance.md)
- [API Reference](docs/api.md)

## Author

Your Name - [GitHub](https://github.com:sohamgugale)
