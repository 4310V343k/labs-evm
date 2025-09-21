#include "parallel.hpp"
#include "../common/common.hpp"
#include <vector>
#include <thread>
#include <future>

namespace integral_parallel {
    namespace {
        double weierstrass_chunk(double a, double b, std::size_t n, double x0, double h, std::size_t begin, std::size_t end) {
            double local = 0.0;
            for (std::size_t i = begin; i < end; ++i)
                local += common::weierstrass(x0 + h * (i + 0.5), a, b, n);
            return local;
        }
    }

    double integrate_weierstrass_parallel(double a, double b, std::size_t n, double x0, double x1, std::size_t steps) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4; // fallback
        // не больше тредов, чем шагов
        std::size_t threads = std::min<std::size_t>(hw, steps);
        // размер куска для 1 треда
        std::size_t chunk = steps / threads;
        // сколько тредов получат на 1 шаг больше
        std::size_t remainder = steps % threads;
        // ширина шага
        double h = (x1 - x0) / static_cast<double>(steps);

        // запускаем треды
        // каждый тред считает свою часть суммы и возвращает результат
        // в главном потоке собираем результаты
        std::vector<std::future<double>> futures;
        std::size_t start = 0;
        for (std::size_t t = 0; t < threads; ++t) {
            // количеству remainder первых тредов даём по 1 дополнительному шагу
            std::size_t size = chunk + (t < remainder ? 1 : 0);
            std::size_t begin = start;
            std::size_t end = begin + size;
            start = end;
            futures.emplace_back(std::async(std::launch::async, weierstrass_chunk, a, b, n, x0, h, begin, end));
        }
        double total = 0.0;
        for (auto &f : futures) total += f.get();
        return total * h;
    }
}
