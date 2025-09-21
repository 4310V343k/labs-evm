use common::{WEIER_A, WEIER_B, WEIER_N, INTEGRAL_X0, INTEGRAL_X1, INTEGRAL_STEPS};
use integral_opencl::integrate_weierstrass_opencl;

fn main() {
    match integrate_weierstrass_opencl(
        WEIER_A,
        WEIER_B,
        WEIER_N,
        INTEGRAL_X0,
        INTEGRAL_X1,
        INTEGRAL_STEPS,
    ) {
        Ok(result) => println!("Integral (OpenCL): {}", result),
        Err(e) => eprintln!("OpenCL error: {:?}", e),
    }
}
