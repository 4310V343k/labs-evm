use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_float, CL_NON_BLOCKING};

const KERNEL_SRC: &str = r#"
__kernel void weier_integral(
    const float a,
    const float b,
    const int n,
    const float x0,
    const float h,
    __global float* out
) {
    int i = get_global_id(0);
    float x = x0 + h * (i + 0.5f);
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += pow(a, (float)k) * cos(3.14159265358979323846f * pow(b, (float)k) * x);
    }
    out[i] = sum;
}
"#;

pub fn integrate_weierstrass_opencl(
    a: f64,
    b: f64,
    n: usize,
    x0: f64,
    x1: f64,
    steps: usize,
) -> Result<f64, String> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);
    let context = Context::from_device(&device)?;
    let queue = CommandQueue::create_default(&context, 0)?;

    let program = Program::create_and_build_from_source(&context, KERNEL_SRC, "")?;
    let kernel = Kernel::create(&program, "weier_integral")?;

    let h = (x1 - x0) / steps as f64;

    let buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, steps, std::ptr::null_mut())
    }?;

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&(a as cl_float))
            .set_arg(&(b as cl_float))
            .set_arg(&(n as cl_float))
            .set_arg(&(x0 as cl_float))
            .set_arg(&(h as cl_float))
            .set_arg(&buffer)
            .set_global_work_size(steps)
            .enqueue_nd_range(&queue)?
    };

    println!("enqueued");
    kernel_event.wait()?;
    println!("finished");

    let mut results: Vec<cl_float> = vec![0.0f32; steps];
    let read_event =
        unsafe { queue.enqueue_read_buffer(&buffer, CL_NON_BLOCKING, 0, &mut results, &[])? };

    read_event.wait()?;

    let integral: f64 = results.iter().map(|&v| v as f64).sum::<f64>() * h;
    Ok(integral)
}
