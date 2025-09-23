#include "mpi_parallel.hpp"
#include "../1_st_mt/cpp/common/common.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>

// Модуль для параллельного численного интегрирования функции Вейерштрасса
// с использованием OpenMPI.
namespace integral_mpi {
    namespace {
        // Функция для вычисления части интеграла на диапазоне [begin, end).
        double weierstrass_chunk(double a, double b, std::size_t n, double x0, double h,
                                std::size_t begin, std::size_t end) {
            double local = 0.0;
            for (std::size_t i = begin; i < end; ++i) {
                // Вычисляем значение функции Вейерштрасса в точке x
                double x = x0 + h * (static_cast<double>(i) + 0.5);
                local += common::weierstrass(x, a, b, n);
            }
            return local;
        }
    }

    // Основная функция для параллельного интегрирования с использованием MPI.
    // a, b, n — параметры функции Вейерштрасса
    // x0, x1 — границы интегрирования
    // steps — количество разбиений (шагов интегрирования)
    double integrate_weierstrass_mpi(double a, double b, std::size_t n,
                                    double x0, double x1, std::size_t steps) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // h — ширина одного шага интегрирования
        double h = (x1 - x0) / static_cast<double>(steps);

        // Распределяем работу между процессами
        // chunk — базовое количество шагов на процесс
        std::size_t chunk = steps / size;
        // remainder — сколько процессов получат на 1 шаг больше (для равномерного распределения)
        std::size_t remainder = steps % size;

        // Определяем диапазон для текущего процесса
        std::size_t local_size = chunk + (rank < remainder ? 1 : 0);
        std::size_t start = rank * chunk + std::min(static_cast<std::size_t>(rank), remainder);
        std::size_t end = start + local_size;

        // Вычисляем локальную часть интеграла
        double local_result = weierstrass_chunk(a, b, n, x0, h, start, end);

        // Собираем результаты от всех процессов
        double global_result = 0.0;
        MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Возвращаем итоговый результат интегрирования
        return global_result * h;
    }
}