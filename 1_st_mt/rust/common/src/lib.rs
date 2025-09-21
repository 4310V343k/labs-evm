pub const WEIER_A: f64 = 0.5;
pub const WEIER_B: f64 = 30.0;
pub const WEIER_N: usize = 20;
pub const INTEGRAL_X0: f64 = 0.0;
pub const INTEGRAL_X1: f64 = 1.0;
pub const INTEGRAL_STEPS: usize = 100_000;

pub fn weierstrass(x: f64, a: f64, b: f64, n: usize) -> f64 {
    let mut sum = 0.0;
    for k in 0..n {
        sum += a.powi(k as i32) * (std::f64::consts::PI * b.powi(k as i32) * x).cos();
    }
    sum
}
