#include "omp_parallel.hpp"
#include "../common/common.hpp"
#include <omp.h>
#include <cmath>

namespace integral_parallel_omp {
    double integrate_weierstrass_parallel_omp(double a, double b, std::size_t n, double x0, double x1, std::size_t steps) {
        double h = (x1 - x0) / static_cast<double>(steps);
        double total = 0.0;
        #pragma omp parallel for reduction(+:total) schedule(static)
        for (long long i = 0; i < static_cast<long long>(steps); ++i) {
            double x = x0 + h * (static_cast<double>(i) + 0.5);
            total += common::weierstrass(x, a, b, n);
        }
        return total * h;
    }
}
