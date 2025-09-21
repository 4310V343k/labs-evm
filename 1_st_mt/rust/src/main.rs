use tabled::{Table, Tabled};
use common::{WEIER_A, WEIER_B, INTEGRAL_X0, INTEGRAL_X1};
use integral_single::integrate_weierstrass;
use integral_parallel::integrate_weierstrass_parallel;
use integral_opencl::integrate_weierstrass_opencl;

#[derive(Tabled)]
struct ResultRow {
    config: String,
    cpu_single: String,
    cpu_parallel: String,
    gpu_opencl: String,
    result_check: String,
}

fn main() {
    let configs = vec![
        (1, 10_000),
        (20, 100_000),
        (30, 1_000_000),
    ];
    let mut rows: Vec<ResultRow> = Vec::new();
    for (n, steps) in configs {
        let a = WEIER_A;
        let b = WEIER_B;
        let x0 = INTEGRAL_X0;
        let x1 = INTEGRAL_X1;

        let t1 = std::time::Instant::now();
        let single_res = integrate_weierstrass(a, b, n, x0, x1, steps);
        let single_time = t1.elapsed().as_secs_f64();

        let t2 = std::time::Instant::now();
        let parallel_res = integrate_weierstrass_parallel(a, b, n, x0, x1, steps);
        let parallel_time = t2.elapsed().as_secs_f64();

        let t3 = std::time::Instant::now();
        let gpu_res = match integrate_weierstrass_opencl(a, b, n, x0, x1, steps) {
            Ok(val) => val,
            Err(e) => {
                eprintln!("OpenCL error: {:?}", e);
                f64::NAN
            }
        };
        let gpu_time = t3.elapsed().as_secs_f64();

        let check = if (single_res - parallel_res).abs() < 1e-6 && (single_res - gpu_res).abs() < 1e-4 {
            "OK"
        } else {
            "FAIL"
        };

        println!("finished ({}) in ", format!("n={}, steps={}", n, steps));

        rows.push(ResultRow {
            config: format!("n={}, steps={}", n, steps),
            cpu_single: format!("{:.6} ({:.3}s)", single_res, single_time),
            cpu_parallel: format!("{:.6} ({:.3}s)", parallel_res, parallel_time),
            gpu_opencl: format!("{:.6} ({:.3}s)", gpu_res, gpu_time),
            result_check: check.to_string(),
        });
    }
    let table = Table::new(rows).to_string();
    println!("\nBenchmark Results:\n{}\n", table);
}
