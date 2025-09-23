#include "mpi_parallel.hpp"
#include "../1_st_mt/cpp/common/common.hpp"
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>

// Simple single-threaded implementation for comparison
double integrate_weierstrass_single(double a, double b, std::size_t n, double x0, double x1, std::size_t steps) {
    double h = (x1 - x0) / static_cast<double>(steps);
    double sum = 0.0;
    for (std::size_t i = 0; i < steps; ++i) {
        double x = x0 + h * (static_cast<double>(i) + 0.5);
        sum += common::weierstrass(x, a, b, n);
    }
    return sum * h;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Список конфигураций для тестирования: (n, steps)
    std::vector<std::pair<int, std::size_t>> configs = {
        {10, 10000},
        {20, 100000},
        {30, 1000000}
    };

    if (rank == 0) {
        std::cout << "MPI Weierstrass Integration Test (with " << size << " processes)\n";
        std::cout << std::left << std::setw(15) << "Config" 
                  << std::setw(20) << "Single Thread" 
                  << std::setw(20) << "MPI Parallel" 
                  << std::setw(15) << "Speedup" 
                  << "Check\n";
        std::cout << std::string(70, '-') << "\n";
    }

    for (const auto &config : configs) {
        int n = config.first;
        std::size_t steps = config.second;
        double a = common::WEIER_A;
        double b = common::WEIER_B;
        double x0 = common::INTEGRAL_X0;
        double x1 = common::INTEGRAL_X1;

        double single_time = 0.0, mpi_time = 0.0;
        double single_result = 0.0, mpi_result = 0.0;

        // Только процесс 0 выполняет однопоточное интегрирование для сравнения
        if (rank == 0) {
            auto t_single = std::chrono::high_resolution_clock::now();
            single_result = integrate_weierstrass_single(a, b, n, x0, x1, steps);
            single_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_single).count();
        }

        // Все процессы участвуют в MPI интегрировании
        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация для точного измерения времени
        auto t_mpi = std::chrono::high_resolution_clock::now();
        mpi_result = integral_mpi::integrate_weierstrass_mpi(a, b, n, x0, x1, steps);
        MPI_Barrier(MPI_COMM_WORLD);
        mpi_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_mpi).count();

        if (rank == 0) {
            bool check = std::abs(single_result - mpi_result) < 1e-6;
            double speedup = single_time / mpi_time;
            
            std::cout << std::left << std::setw(15) << ("n=" + std::to_string(n) + ",s=" + std::to_string(steps/1000) + "k")
                      << std::setw(20) << (std::to_string(single_result).substr(0,8) + " (" + std::to_string(single_time).substr(0,6) + "s)")
                      << std::setw(20) << (std::to_string(mpi_result).substr(0,8) + " (" + std::to_string(mpi_time).substr(0,6) + "s)")
                      << std::setw(15) << (std::to_string(speedup).substr(0,5) + "x")
                      << (check ? "OK" : "FAIL") << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}