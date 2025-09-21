#include "single.hpp"
#include "../common/common.hpp"

namespace integral_single {
    double integrate_weierstrass(double a, double b, std::size_t n, double x0, double x1, std::size_t steps) {
        double h = (x1 - x0) / static_cast<double>(steps);
        double sum = 0.0;
        for (std::size_t i = 0; i < steps; ++i) {
            double x = x0 + h * (static_cast<double>(i) + 0.5);
            sum += common::weierstrass(x, a, b, n);
        }
        return sum * h;
    }
}
