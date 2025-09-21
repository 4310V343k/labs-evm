#include "parallel.hpp"
#include "../common/common.hpp"
#include <vector>
#include <thread>
#include <thread>

namespace integral_parallel {
    namespace {
        void weierstrass_chunk(double a, double b, std::size_t n, double x0, double h, std::size_t begin, std::size_t end, double* out) {
            double local = 0.0;
            for (std::size_t i = begin; i < end; ++i)
                local += common::weierstrass(x0 + h * (i + 0.5), a, b, n);
            *out = local;
        }
    }

    double integrate_weierstrass_parallel(double a, double b, std::size_t n, double x0, double x1, std::size_t steps) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4; // fallback
        std::size_t threads = std::min<std::size_t>(hw, steps);
        std::size_t chunk = steps / threads;
        std::size_t remainder = steps % threads;
        double h = (x1 - x0) / static_cast<double>(steps);

        std::vector<double> results(threads, 0.0);
        std::vector<std::thread> workers;
        std::size_t start = 0;
        for (std::size_t t = 0; t < threads; ++t) {
            std::size_t size = chunk + (t < remainder ? 1 : 0);
            std::size_t begin = start;
            std::size_t end = begin + size;
            start = end;
            workers.emplace_back(weierstrass_chunk, a, b, n, x0, h, begin, end, &results[t]);
        }
        for (auto& th : workers) th.join();
        double total = 0.0;
        for (double v : results) total += v;
        return total * h;
    }
}
