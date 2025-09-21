#pragma once
#include <cstddef>

namespace integral_single {
    double integrate_weierstrass(double a, double b, std::size_t n, double x0, double x1, std::size_t steps);
}
