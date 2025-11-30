/// Comprehensive tensor operation benchmarking tool
/// 
/// Benchmarks the following operations across various tensor sizes:
/// - Memory transfer: CPU ↔ GPU
/// - Elementwise operations: add, subtract, multiply (CPU and GPU)
/// - Different data types: i32, f32, f64
/// - Various tensor shapes: 1D, 2D, 3D, large tensors
/// 
/// Results are exported to a CSV file for analysis.

use std::time::{Duration, Instant};
use csv::Writer;
use serde::Serialize;
use tensors::core::{
    primitives::{CudaTensor, CpuTensor},
    value::{TensorValue, TensorValueElementwise},
    tensor::AsViewMut,
    MetaTensorView,
};

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    operation: String,
    backend: String,
    data_type: String,
    shape: String,
    size: usize,
    duration_ms: f64,
    throughput_gb_s: f64,
}

impl BenchmarkResult {
    fn new(
        operation: &str,
        backend: &str,
        data_type: &str,
        shape: &str,
        size: usize,
        duration: Duration,
        bytes: usize,
    ) -> Self {
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let throughput_gb_s = if duration_ms > 0.0 {
            (bytes as f64 / 1_000_000_000.0) / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            operation: operation.to_string(),
            backend: backend.to_string(),
            data_type: data_type.to_string(),
            shape: shape.to_string(),
            size,
            duration_ms,
            throughput_gb_s,
        }
    }
}

/// Benchmark memory transfer from CPU to GPU
fn bench_cpu_to_gpu<T: TensorValue>(
    cpu_tensor: &CpuTensor<T>,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cpu_tensor.size();
    let bytes = size * std::mem::size_of::<T>();

    let start = Instant::now();
    let _cuda_tensor = cpu_tensor.cuda().unwrap();
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "copy_cpu_to_gpu",
        "cuda",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark memory transfer from GPU to CPU
fn bench_gpu_to_cpu<T: TensorValue>(
    cuda_tensor: &CudaTensor<T>,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cuda_tensor.size();
    let bytes = size * std::mem::size_of::<T>();

    let start = Instant::now();
    let _cpu_tensor = cuda_tensor.cpu().unwrap();
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "copy_gpu_to_cpu",
        "cpu",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark addition on CPU
fn bench_add_cpu<T: TensorValue + TensorValueElementwise + std::ops::AddAssign>(
    cpu_tensor: &mut CpuTensor<T>,
    value: T,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cpu_tensor.size();
    let bytes = size * std::mem::size_of::<T>() * 2; // read + write

    let start = Instant::now();
    {
        let mut view = cpu_tensor.view_mut();
        view += value;
    }
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "add",
        "cpu",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark addition on GPU
fn bench_add_gpu<T: TensorValue + TensorValueElementwise>(
    cuda_tensor: &mut CudaTensor<T>,
    value: T,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cuda_tensor.size();
    let bytes = size * std::mem::size_of::<T>() * 2; // read + write

    let start = Instant::now();
    {
        let mut view = cuda_tensor.view_mut();
        view += value;
    }
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "add",
        "cuda",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark subtraction on CPU
fn bench_sub_cpu<T: TensorValue + TensorValueElementwise + std::ops::SubAssign>(
    cpu_tensor: &mut CpuTensor<T>,
    value: T,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cpu_tensor.size();
    let bytes = size * std::mem::size_of::<T>() * 2;

    let start = Instant::now();
    {
        let mut view = cpu_tensor.view_mut();
        view -= value;
    }
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "subtract",
        "cpu",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark subtraction on GPU
fn bench_sub_gpu<T: TensorValue + TensorValueElementwise>(
    cuda_tensor: &mut CudaTensor<T>,
    value: T,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cuda_tensor.size();
    let bytes = size * std::mem::size_of::<T>() * 2;

    let start = Instant::now();

    let mut view = cuda_tensor.view_mut();
    view -= value;
    
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "subtract",
        "cuda",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark multiplication on CPU
fn bench_mul_cpu<T: TensorValue + TensorValueElementwise + std::ops::MulAssign>(
    cpu_tensor: &mut CpuTensor<T>,
    value: T,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cpu_tensor.size();
    let bytes = size * std::mem::size_of::<T>() * 2;

    let start = Instant::now();
    {
        let mut view = cpu_tensor.view_mut();
        view *= value;
    }
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "multiply",
        "cpu",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Benchmark multiplication on GPU
fn bench_mul_gpu<T: TensorValue + TensorValueElementwise + std::ops::MulAssign>(
    cuda_tensor: &mut CudaTensor<T>,
    value: T,
    shape_str: &str,
    type_name: &str,
    results: &mut Vec<BenchmarkResult>,
) {
    let size = cuda_tensor.size();
    let bytes = size * std::mem::size_of::<T>() * 2;

    let start = Instant::now();
    {
        let mut view = cuda_tensor.view_mut();
        view *= value;
    }
    let duration = start.elapsed();

    results.push(BenchmarkResult::new(
        "multiply",
        "cuda",
        type_name,
        shape_str,
        size,
        duration,
        bytes,
    ));
}

/// Run comprehensive benchmarks for a specific data type
fn run_benchmarks_for_type<T>(
    data: Vec<T>,
    shape: Vec<usize>,
    shape_str: &str,
    type_name: &str,
    add_val: T,
    sub_val: T,
    mul_val: T,
    results: &mut Vec<BenchmarkResult>,
)
where
    T: TensorValue + TensorValueElementwise + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign,
{
    println!("  Benchmarking {} tensor with shape {}", type_name, shape_str);

    // Create CPU tensor
    let mut cpu_tensor = CpuTensor::<T>::from_buf(data.clone(), shape.clone()).unwrap();

    // Benchmark CPU to GPU transfer
    bench_cpu_to_gpu(&cpu_tensor, shape_str, type_name, results);

    // Create GPU tensor
    let mut cuda_tensor = cpu_tensor.cuda().unwrap();

    // Benchmark GPU to CPU transfer
    bench_gpu_to_cpu(&cuda_tensor, shape_str, type_name, results);

    // Benchmark CPU operations
    bench_add_cpu(&mut cpu_tensor, add_val, shape_str, type_name, results);
    bench_sub_cpu(&mut cpu_tensor, sub_val, shape_str, type_name, results);
    bench_mul_cpu(&mut cpu_tensor, mul_val, shape_str, type_name, results);

    // Benchmark GPU operations
    bench_add_gpu(&mut cuda_tensor, add_val, shape_str, type_name, results);
    bench_sub_gpu(&mut cuda_tensor, sub_val, shape_str, type_name, results);
    bench_mul_gpu(&mut cuda_tensor, mul_val, shape_str, type_name, results);
}

fn main() {
    println!("Starting tensor operation benchmarks...\n");

    let mut results = Vec::new();

    // Define test configurations: (shape, shape_string)
    let test_configs = vec![
        (vec![1000], "1D-1K"),
        (vec![10_000], "1D-10K"),
        (vec![100_000], "1D-100K"),
        (vec![1_000_000], "1D-1M"),
        (vec![10_000_000], "1D-10M"),
        (vec![100, 100], "2D-100x100"),
        (vec![1000, 1000], "2D-1Kx1K"),
        (vec![3000, 3000], "2D-3Kx3K"),
        (vec![10, 10, 10], "3D-10x10x10"),
        (vec![100, 100, 100], "3D-100x100x100"),
        (vec![4, 512, 512], "3D-4x512x512"),
        (vec![512, 512, 512], "3D-512x512x512"),
        (vec![512, 512, 1024], "3D-512x512x1024"),
    ];

    for (shape, shape_str) in test_configs {
        let size: usize = shape.iter().product();
        println!("Running benchmarks for shape: {} (size: {})", shape_str, size);

        // i32 benchmarks
        {
            let data: Vec<i32> = (0..size).map(|i| (i % 100) as i32).collect();
            run_benchmarks_for_type(
                data,
                shape.clone(),
                shape_str,
                "i32",
                5,
                3,
                2,
                &mut results,
            );
        }

        // f32 benchmarks
        {
            let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32 * 0.1).collect();
            run_benchmarks_for_type(
                data,
                shape.clone(),
                shape_str,
                "f32",
                1.5,
                0.5,
                2.0,
                &mut results,
            );
        }

        // f64 benchmarks
        {
            let data: Vec<f64> = (0..size).map(|i| (i % 100) as f64 * 0.01).collect();
            run_benchmarks_for_type(
                data,
                shape.clone(),
                shape_str,
                "f64",
                0.75,
                0.25,
                1.5,
                &mut results,
            );
        }

        println!();
    }

    // Write results to CSV
    let csv_path = "benchmark_results.csv";
    println!("Writing results to {}...", csv_path);

    let mut writer = Writer::from_path(csv_path).expect("Failed to create CSV file");

    for result in &results {
        writer.serialize(result).expect("Failed to write record");
    }

    writer.flush().expect("Failed to flush CSV writer");

    println!("\nBenchmark complete!");
    println!("Total benchmarks run: {}", results.len());
    println!("Results saved to: {}", csv_path);

    // Print summary statistics
    println!("\n=== Summary Statistics ===");
    
    // Group by operation and backend
    let mut cpu_times: Vec<f64> = Vec::new();
    let mut gpu_times: Vec<f64> = Vec::new();
    
    for result in &results {
        if result.backend == "cpu" && result.operation != "copy_gpu_to_cpu" {
            cpu_times.push(result.duration_ms);
        } else if result.backend == "cuda" && result.operation != "copy_cpu_to_gpu" {
            gpu_times.push(result.duration_ms);
        }
    }
    
    if !cpu_times.is_empty() {
        let cpu_avg = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
        println!("Average CPU operation time: {:.3} ms", cpu_avg);
    }
    
    if !gpu_times.is_empty() {
        let gpu_avg = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
        println!("Average GPU operation time: {:.3} ms", gpu_avg);
        
        if !cpu_times.is_empty() {
            let cpu_avg = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
            let speedup = cpu_avg / gpu_avg;
            println!("Average GPU speedup: {:.2}x", speedup);
        }
    }
}
