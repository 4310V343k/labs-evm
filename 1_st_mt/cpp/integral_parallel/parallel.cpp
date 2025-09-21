#include "parallel.hpp"
#include "../common/common.hpp"
#include <vector>
#include <thread>
#include <thread>


// Модуль для параллельного численного интегрирования функции Вейерштрасса
// с использованием стандартных потоков C++ (std::thread).
namespace integral_parallel {
    namespace {
        // Функция для вычисления части интеграла на диапазоне [begin, end).
        // Результат записывается по адресу out.
        void weierstrass_chunk(double a, double b, std::size_t n, double x0, double h,
                              std::size_t begin, std::size_t end, double* out) {
            double local = 0.0;
            for (std::size_t i = begin; i < end; ++i) {
                // Вычисляем значение функции Вейерштрасса в точке x
                double x = x0 + h * (static_cast<double>(i) + 0.5);
                local += common::weierstrass(x, a, b, n);
            }
            *out = local;
        }
    }

    // Основная функция для параллельного интегрирования.
    // a, b, n — параметры функции Вейерштрасса
    // x0, x1 — границы интегрирования
    // steps — количество разбиений (шагов интегрирования)
    double integrate_weierstrass_parallel(double a, double b, std::size_t n,
                                          double x0, double x1, std::size_t steps) {
        // hw — количество аппаратных потоков (ядер/логических процессоров)
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4; // если не удалось определить, используем 4 потока

        // threads — итоговое количество потоков (не больше, чем шагов)
        std::size_t threads = std::min<std::size_t>(hw, steps);

        // chunk — базовое количество шагов на поток
        std::size_t chunk = steps / threads;
        // remainder — сколько потоков получат на 1 шаг больше (для равномерного распределения)
        std::size_t remainder = steps % threads;

        // h — ширина одного шага интегрирования
        double h = (x1 - x0) / static_cast<double>(steps);

        // results — массив для хранения результата каждого потока
        std::vector<double> results(threads, 0.0);
        // workers — массив потоков
        std::vector<std::thread> workers;

        // start — индекс первого шага для текущего потока
        std::size_t start = 0;
        for (std::size_t t = 0; t < threads; ++t) {
            // size — количество шагов для текущего потока
            std::size_t size = chunk + (t < remainder ? 1 : 0);
            std::size_t begin = start;
            std::size_t end = begin + size;
            start = end;
            // Запускаем поток, вычисляющий свою часть интеграла
            workers.emplace_back(weierstrass_chunk, a, b, n, x0, h, begin, end, &results[t]);
        }
        // Ожидаем завершения всех потоков
        for (auto& th : workers) th.join();

        // Суммируем результаты всех потоков
        double total = 0.0;
        for (double v : results) total += v;

        // Возвращаем итоговый результат интегрирования
        return total * h;
    }
}
