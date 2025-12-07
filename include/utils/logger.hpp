#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace mg {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }
    
    void set_level(LogLevel level) { level_ = level; }
    void set_file(const std::string& filename) {
        file_.open(filename, std::ios::out);
    }
    
    template<typename... Args>
    void debug(Args&&... args) {
        log(LogLevel::DEBUG, "DEBUG", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void info(Args&&... args) {
        log(LogLevel::INFO, "INFO", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void warning(Args&&... args) {
        log(LogLevel::WARNING, "WARNING", std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void error(Args&&... args) {
        log(LogLevel::ERROR, "ERROR", std::forward<Args>(args)...);
    }
    
private:
    Logger() : level_(LogLevel::INFO) {}
    
    template<typename... Args>
    void log(LogLevel msg_level, const char* prefix, Args&&... args) {
        if (msg_level < level_) return;
        
        std::ostringstream oss;
        oss << "[" << prefix << "] ";
        ((oss << std::forward<Args>(args)), ...);
        
        std::string msg = oss.str();
        std::cout << msg << std::endl;
        
        if (file_.is_open()) {
            file_ << msg << std::endl;
        }
    }
    
    LogLevel level_;
    std::ofstream file_;
};

} // namespace mg
