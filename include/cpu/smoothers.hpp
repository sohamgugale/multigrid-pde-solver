#pragma once
#include "common/matrix.hpp"
#include <memory>

namespace mg {

enum class SmootherType {
    JACOBI,
    GAUSS_SEIDEL,
    RED_BLACK_GS
};

class Smoother {
public:
    virtual ~Smoother() = default;
    
    // Perform nu smoothing iterations on Au = b
    virtual void smooth(const SparseMatrix& A, Vector& u, const Vector& b, 
                       int nu) = 0;
    
    // Factory method
    static std::unique_ptr<Smoother> create(SmootherType type, double omega = 1.0);
};

class JacobiSmoother : public Smoother {
public:
    JacobiSmoother(double omega = 0.67) : omega_(omega) {}
    
    void smooth(const SparseMatrix& A, Vector& u, const Vector& b, int nu) override;
    
private:
    double omega_;  // Relaxation parameter
};

class GaussSeidelSmoother : public Smoother {
public:
    GaussSeidelSmoother(double omega = 1.0) : omega_(omega) {}
    
    void smooth(const SparseMatrix& A, Vector& u, const Vector& b, int nu) override;
    
private:
    double omega_;
};

} // namespace mg
