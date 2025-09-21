use common::weierstrass;

/// Однопоточная реализация интегрирования функции Вейерштрасса
pub fn integrate_weierstrass(a: f64, b: f64, n: usize, x0: f64, x1: f64, steps: usize) -> f64 {
    let h = (x1 - x0) / steps as f64;
    let sum: f64 = (0..steps)
        .map(|i| {
            let x = x0 + h * (i as f64 + 0.5);
            weierstrass(x, a, b, n)
        })
        .sum();
    sum * h
}
