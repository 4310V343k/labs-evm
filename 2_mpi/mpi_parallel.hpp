#pragma once
#include <cstddef>

namespace integral_mpi {
    double integrate_weierstrass_mpi(double a, double b, std::size_t n, double x0, double x1, std::size_t steps);
}