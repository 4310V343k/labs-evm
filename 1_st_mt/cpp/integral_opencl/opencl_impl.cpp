#include "opencl_impl.hpp"
#include "../common/common.hpp"
#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <mutex>

// Модуль для численного интегрирования функции Вейерштрасса на GPU с помощью OpenCL
namespace integral_opencl {
    namespace {
        // Глобальная переменная для хранения последней ошибки OpenCL
        std::string g_last_error;
        std::mutex g_err_mutex;

        // Исходный код ядра OpenCL для вычисления функции Вейерштрасса
        const char *kernelSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void weier_integral(
    const double a,
    const double b,
    const int n,
    const double x0,
    const double h,
    __global double* out
) {
    int i = get_global_id(0);
    double x = x0 + h * (i + 0.5);
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += pow(a, (double)k) * cos(3.14159265358979323846 * pow(b, (double)k) * x);
    }
    out[i] = sum;
}
)CLC";

        // Устанавливает текст ошибки для последующего получения
        void set_error(const std::string &msg) {
            std::lock_guard<std::mutex> lock(g_err_mutex);
            g_last_error = msg;
        }
    }


    // Возвращает текст последней ошибки OpenCL
    const std::string & last_error() { return g_last_error; }

    // Основная функция для интегрирования функции Вейерштрасса на GPU через OpenCL
    // a, b, n — параметры функции
    // x0, x1 — границы интегрирования
    // steps — количество разбиений
    // ok — флаг успешности вычисления
    double integrate_weierstrass_opencl(double a, double b, std::size_t n, double x0, double x1, std::size_t steps, bool &ok) {
        ok = false;

        // Получаем список доступных платформ OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) { set_error("No OpenCL platforms"); return 0.0; }

        // Получаем список устройств (GPU) на первой платформе
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) { set_error("No GPU devices"); return 0.0; }

        // Выбираем первое устройство и создаём контекст и очередь команд
        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // Создаём и компилируем программу OpenCL из исходного кода ядра
        cl::Program::Sources sources;
        sources.push_back({kernelSrc, std::strlen(kernelSrc)});
        cl::Program program(context, sources);
        if (program.build({device}) != CL_SUCCESS) {
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            set_error(std::string("Build error: ") + log);
            return 0.0;
        }

        // Создаём объект ядра
        cl::Kernel kernel(program, "weier_integral");

        // Вычисляем ширину шага интегрирования
        double h = (x1 - x0) / static_cast<double>(steps);
        // Буфер для хранения результатов вычислений на GPU
        cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY, sizeof(double) * steps);

        // Подготавливаем аргументы для ядра
        double ad = a;
        double bd = b;
        int   nd = static_cast<int>(n);
        double x0d = x0;
        double hd = h;

        // Устанавливаем аргументы ядра
        kernel.setArg(0, ad);
        kernel.setArg(1, bd);
        kernel.setArg(2, nd);
        kernel.setArg(3, x0d);
        kernel.setArg(4, hd);
        kernel.setArg(5, outBuf);

        // Запускаем ядро на GPU
        cl::NDRange globalWorkSize(steps);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize);
        queue.finish();

        // Читаем результаты вычислений из буфера
        std::vector<double> results(steps, 0.0);
        queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(double) * steps, results.data());

        // Суммируем значения и вычисляем интеграл
        double sum = 0.0; for (double v : results) sum += v;
        double integral = sum * h;
        ok = true;
        return integral;
    }
}
