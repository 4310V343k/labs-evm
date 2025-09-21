#include "opencl_impl.hpp"
#include "../common/common.hpp"
#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <mutex>

namespace integral_opencl {
    namespace {
        std::string g_last_error;
        std::mutex g_err_mutex;
        const char *kernelSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void weier_integral_reduce(
    const double a,
    const double b,
    const int n,
    const double x0,
    const double h,
    const ulong steps,
    __global double* out,
    __local double* lsum
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);
    int group_size = get_local_size(0);

    double val = 0.0;
    if ((ulong)gid < steps) {
        double x = x0 + h * (gid + 0.5);
        for (int k = 0; k < n; ++k) {
            val += pow(a, (double)k) * cos(3.14159265358979323846 * pow(b, (double)k) * x);
        }
    }
    lsum[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (lid < offset) {
            lsum[lid] += lsum[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[group] = lsum[0];
    }
}
)CLC";
        void set_error(const std::string &msg) {
            std::lock_guard<std::mutex> lock(g_err_mutex);
            g_last_error = msg;
        }
    }

    const std::string & last_error() { return g_last_error; }

    double integrate_weierstrass_opencl(double a, double b, std::size_t n, double x0, double x1, std::size_t steps, bool &ok) {
        ok = false;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) { set_error("No OpenCL platforms"); return 0.0; }

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) { set_error("No GPU devices"); return 0.0; }

        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Program::Sources sources;
        sources.push_back({kernelSrc, std::strlen(kernelSrc)});
        cl::Program program(context, sources);
        if (program.build({device}) != CL_SUCCESS) {
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            set_error(std::string("Build error: ") + log);
            return 0.0;
        }

        double h = (x1 - x0) / static_cast<double>(steps);
        size_t local_size = 256;
        size_t num_groups = (steps + local_size - 1) / local_size;
        size_t global_size = num_groups * local_size;

        cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_groups);

        double ad = a;
        double bd = b;
        int   nd = static_cast<int>(n);
        double x0d = x0;
        double hd = h;

        cl::Kernel kernel(program, "weier_integral_reduce");
        kernel.setArg(0, ad);
        kernel.setArg(1, bd);
        kernel.setArg(2, nd);
        kernel.setArg(3, x0d);
        kernel.setArg(4, hd);
        kernel.setArg(5, static_cast<cl_ulong>(steps));
        kernel.setArg(6, outBuf);
        kernel.setArg(7, cl::Local(local_size * sizeof(double)));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
        queue.finish();

        std::vector<double> results(num_groups, 0.0);
        queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(double) * num_groups, results.data());

        double sum = 0.0;
        for (double v : results) sum += v;
        double integral = sum * h;
        ok = true;
        return integral;
    }
}
