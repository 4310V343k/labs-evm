use common::{WEIER_A, WEIER_B, WEIER_N, INTEGRAL_X0, INTEGRAL_X1, INTEGRAL_STEPS};
use integral_parallel::integrate_weierstrass_parallel;

fn main() {
    let result = integrate_weierstrass_parallel(
        WEIER_A,
        WEIER_B,
        WEIER_N,
        INTEGRAL_X0,
        INTEGRAL_X1,
        INTEGRAL_STEPS,
    );
    println!("Integral (multi-threaded): {}", result);
}
