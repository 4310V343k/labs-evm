#include "common.hpp"
#include <cmath>

namespace common {
    double weierstrass(double x, double a, double b, std::size_t n) {
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            sum += std::pow(a, static_cast<double>(k)) * std::cos(PI * std::pow(b, static_cast<double>(k)) * x);
        }
        return sum;
    }
}
