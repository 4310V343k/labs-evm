use common::{WEIER_A, WEIER_B, WEIER_N, INTEGRAL_X0, INTEGRAL_X1, INTEGRAL_STEPS};
use integral_single::integrate_weierstrass;

fn main() {
    let result = integrate_weierstrass(
        WEIER_A,
        WEIER_B,
        WEIER_N,
        INTEGRAL_X0,
        INTEGRAL_X1,
        INTEGRAL_STEPS,
    );
    println!("Integral (single-thread): {}", result);
}
