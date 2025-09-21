#pragma once
#include <cstddef>

namespace common {
    static constexpr double WEIER_A = 0.5;
    static constexpr double WEIER_B = 30.0;
    static constexpr double INTEGRAL_X0 = 0.0;
    static constexpr double INTEGRAL_X1 = 1.0;

    static constexpr double PI = 3.141592653589793238462643383279502884;

    double weierstrass(double x, double a, double b, std::size_t n);
}
