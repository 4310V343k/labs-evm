#pragma once
#include <cstddef>

namespace integral_parallel_omp {
    double integrate_weierstrass_parallel_omp(double a, double b, std::size_t n, double x0, double x1, std::size_t steps);
}
