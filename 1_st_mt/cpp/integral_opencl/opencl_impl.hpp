#pragma once
#include <cstddef>
#include <string>

namespace integral_opencl {
    // Returns integral value on success, std::nullopt on error; error message in last_error()
    double integrate_weierstrass_opencl(double a, double b, std::size_t n, double x0, double x1, std::size_t steps, bool &ok);
    const std::string & last_error();
}
