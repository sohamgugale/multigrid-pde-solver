#pragma once
#include <chrono>
#include <string>
#include <iostream>

namespace mg {

class Timer {
public:
    Timer() { reset(); }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
    
    // Print elapsed time with label
    void print(const std::string& label) const {
        std::cout << label << ": " << elapsed() << " seconds" << std::endl;
    }
    
    // Get current timestamp
    static double now() {
        static auto program_start = std::chrono::high_resolution_clock::now();
        auto current = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(current - program_start).count();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// RAII timer that prints on destruction
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name), timer_() {}
    
    ~ScopedTimer() {
        std::cout << name_ << ": " << timer_.elapsed() << " seconds" << std::endl;
    }
    
private:
    std::string name_;
    Timer timer_;
};

} // namespace mg
