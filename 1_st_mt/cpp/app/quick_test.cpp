#include "../common/common.hpp"
#include "../integral_single/single.hpp"
#include "../integral_parallel/parallel.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    double a = common::WEIER_A;
    double b = common::WEIER_B;
    std::size_t n = 5;
    double x0 = common::INTEGRAL_X0;
    double x1 = common::INTEGRAL_X1;
    std::size_t steps = 10000;

    double s = integral_single::integrate_weierstrass(a,b,n,x0,x1,steps);
    double p = integral_parallel::integrate_weierstrass_parallel(a,b,n,x0,x1,steps);
    if (std::abs(s-p) > 1e-6) {
        std::cerr << "Mismatch single vs parallel: " << s << " vs " << p << "\n";
        return 1;
    }
    std::cout << "OK single vs parallel: " << s << "\n";
    return 0;
}
