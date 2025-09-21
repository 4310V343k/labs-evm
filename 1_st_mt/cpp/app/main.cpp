#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <limits>
#include <string>
#include "../common/common.hpp"
#include "../integral_single/single.hpp"
#include "../integral_parallel/parallel.hpp"
#ifdef ENABLE_OPENMP
#include "../integral_parallel_omp/omp_parallel.hpp"
#endif
#ifdef ENABLE_OPENCL
#include "../integral_opencl/opencl_impl.hpp"
#endif

struct ResultRow {
    std::string config;
    std::string cpu_single;
    std::string cpu_parallel;
    std::string cpu_openmp;
    std::string gpu_opencl;
    std::string result_check;
};

int main() {
    std::vector<std::pair<int,std::size_t>> configs = {
        {10, 10000},
        {20, 100000},
        {30, 1000000},
        {30, 10000000},
        {30, 100000000}
    };
    std::vector<ResultRow> rows;

    for (std::size_t idx = 0; idx < configs.size(); ++idx) {
        int n = configs[idx].first;
        std::size_t steps = configs[idx].second;
        double a = common::WEIER_A;
        double b = common::WEIER_B;
        double x0 = common::INTEGRAL_X0;
        double x1 = common::INTEGRAL_X1;

        auto t_single = std::chrono::high_resolution_clock::now();
        double single_res = integral_single::integrate_weierstrass(a, b, n, x0, x1, steps);
        double single_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t_single).count();

        auto t_parallel = std::chrono::high_resolution_clock::now();
        double parallel_res = integral_parallel::integrate_weierstrass_parallel(a, b, n, x0, x1, steps);
        double parallel_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t_parallel).count();

        double openmp_res = 0.0; double openmp_time = 0.0; bool openmp_available = false;
        double gpu_res = 0.0; double gpu_time = 0.0; bool gpu_ok = false;
#ifdef ENABLE_OPENMP
        {
            auto t_omp = std::chrono::high_resolution_clock::now();
            openmp_res = integral_parallel_omp::integrate_weierstrass_parallel_omp(a, b, n, x0, x1, steps);
            openmp_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t_omp).count();
            openmp_available = true;
        }
#endif
#ifdef ENABLE_OPENCL
        {
            auto t_gpu = std::chrono::high_resolution_clock::now();
            gpu_res = integral_opencl::integrate_weierstrass_opencl(a, b, n, x0, x1, steps, gpu_ok);
            gpu_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t_gpu).count();
            if (!gpu_ok) {
                std::cerr << "OpenCL error: " << integral_opencl::last_error() << "\n";
                gpu_res = std::numeric_limits<double>::quiet_NaN();
            }
        }
#endif

        bool check_parallel = std::abs(single_res - parallel_res) < 1e-6;
        bool check_openmp = std::abs(single_res - openmp_res) < 1e-6;
        bool check_gpu = std::abs(single_res - gpu_res) < 1e-6;
        if (!check_parallel) {
            std::cerr << "[WARN] CPU parallel mismatch: |single - parallel| = "
                    << std::abs(single_res - parallel_res) << " > 1e-6\n";
        }
        if (openmp_available && !check_openmp) {
            std::cerr << "[WARN] OpenMP mismatch: |single - openmp| = "
                    << std::abs(single_res - openmp_res) << " > 1e-6\n";
        }
        if (gpu_ok && !check_gpu) {
            if (std::isnan(gpu_res) || std::isinf(gpu_res)) {
                std::cerr << "[WARN] GPU result invalid (NaN/Inf)\n";
            } else {
                std::cerr << "[WARN] GPU mismatch: |single - gpu| = " << std::abs(single_res - gpu_res) << " > 1e-6\n";
            }
        }

        bool check = check_parallel && (!openmp_available || check_openmp) && (!gpu_ok || check_gpu);
        std::cout << "finished (n=" << n << ", steps=" << steps << ")\n";

        std::ostringstream ssSingle, ssParallel, ssOmp, ssGpu;
        ssSingle << std::fixed << std::setprecision(6) << single_res << " (" << std::setprecision(3) << single_time << "s)";
        ssParallel << std::fixed << std::setprecision(6) << parallel_res << " (" << std::setprecision(3) << parallel_time << "s)";
        if (openmp_available)
            ssOmp << std::fixed << std::setprecision(6) << openmp_res << " (" << std::setprecision(3) << openmp_time << "s)";
        else
            ssOmp << "N/A";
        if (gpu_ok)
            ssGpu << std::fixed << std::setprecision(6) << gpu_res << " (" << std::setprecision(3) << gpu_time << "s)";
        else
            ssGpu << "N/A";

        std::ostringstream cfg;
        cfg << "n=" << n << ", steps=" << steps;
        ResultRow row;
        row.config = cfg.str();
        row.cpu_single = ssSingle.str();
        row.cpu_parallel = ssParallel.str();
        row.cpu_openmp = ssOmp.str();
        row.gpu_opencl = ssGpu.str();
        row.result_check = check ? "OK" : "FAIL";
        rows.push_back(row);
    }

    std::cout << "\nBenchmark Results:\n";
    std::cout << std::left << std::setw(25) << "Config" << std::setw(30) << "CPU Single" << std::setw(30) << "CPU Parallel" << std::setw(30) << "CPU OpenMP" << std::setw(30) << "GPU OpenCL" << "Check" << "\n";
    for (const auto &r : rows) {
        std::cout << std::left << std::setw(25) << r.config << std::setw(30) << r.cpu_single << std::setw(30) << r.cpu_parallel << std::setw(30) << r.cpu_openmp << std::setw(30) << r.gpu_opencl << r.result_check << "\n";
    }
    std::cout << "\n";
}
