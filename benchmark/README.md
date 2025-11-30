# Tensor Operations Benchmark Suite

This benchmark suite provides comprehensive performance testing for tensor operations on both CPU and GPU backends.

## Overview

The benchmark measures:
- **Memory transfer**: CPU ↔ GPU data movement
- **Elementwise operations**: Addition, subtraction, multiplication
- **Data types**: i32, f32, f64
- **Tensor shapes**: 1D, 2D, 3D with various sizes

## Usage

### Running Benchmarks

```bash
# Build and run the benchmark (release mode for accurate measurements)
cargo run --release

# Or build first, then run
cargo build --release
./target/release/benchmark
```

The benchmark will:
1. Run comprehensive tests across different tensor sizes and shapes
2. Generate `benchmark_results.csv` with detailed metrics
3. Print summary statistics to the console

### Visualizing Results

First, ensure you have the required Python dependencies:

```bash
pip install pandas matplotlib seaborn numpy
```

Then run the visualization script:

```bash
python visualize.py
```

This will generate:
- `plots/cpu_vs_gpu_comparison.png` - Performance comparison charts
- `plots/memory_transfer.png` - Memory transfer analysis
- `plots/scaling_analysis.png` - Scaling behavior across tensor sizes
- `plots/data_type_comparison.png` - Performance by data type
- `plots/performance_heatmap.png` - Heatmap visualizations
- `plots/summary_statistics.txt` - Detailed statistics summary

## Output Files

### benchmark_results.csv

CSV format with the following columns:
- `operation`: Operation type (add, subtract, multiply, copy_cpu_to_gpu, copy_gpu_to_cpu)
- `backend`: Backend used (cpu, cuda)
- `data_type`: Data type (i32, f32, f64)
- `shape`: Tensor shape descriptor (e.g., "1D-1M", "2D-1Kx1K")
- `size`: Total number of elements
- `duration_ms`: Operation duration in milliseconds
- `throughput_gb_s`: Throughput in gigabytes per second

### plots/

Directory containing all generated visualizations and summary statistics.

## Test Configurations

The benchmark tests the following tensor shapes:
- 1D: 1K, 10K, 100K, 1M, 10M elements
- 2D: 100×100, 1K×1K, 3K×3K
- 3D: 10×10×10, 100×100×100, 4×512×512

For each shape, it tests:
- All three data types (i32, f32, f64)
- Both CPU and GPU backends
- All supported operations

## Performance Metrics

### Duration
Raw execution time in milliseconds. Lower is better.

### Throughput
Data processing rate in GB/s, calculated as:
```
throughput = (bytes_processed) / (duration_seconds) / 1_000_000_000
```

For elementwise operations, bytes_processed includes both read and write operations.

## Interpreting Results

### CPU vs GPU
- GPU typically shows advantage for larger tensors (>100K elements)
- Smaller tensors may be faster on CPU due to kernel launch overhead
- Memory transfer overhead can offset GPU computation gains

### Data Types
- Smaller data types (i32) generally show higher throughput
- Floating-point operations (f32, f64) may have different performance characteristics
- GPU often shows better relative performance for floating-point operations

### Scaling
- Check the scaling plots to understand how performance changes with tensor size
- Look for linear scaling in throughput as size increases
- Identify sweet spots for optimal GPU utilization

## Troubleshooting

### CUDA not available
If you see errors about CUDA not being available:
- Ensure you have a CUDA-capable GPU
- Verify CUDA toolkit is installed
- Check that the `cuda` feature is enabled in Cargo.toml

### Out of memory errors
For very large tensors:
- Reduce the maximum tensor sizes in `main.rs`
- Monitor GPU memory usage with `nvidia-smi`

## Extending the Benchmark

To add new operations or configurations:

1. Add new test configurations in `main.rs`:
```rust
let test_configs = vec![
    // Add your custom shapes here
    (vec![your_shape], "description"),
];
```

2. Implement benchmark functions for new operations following the existing pattern

3. Update `visualize.py` to include visualization for new metrics

## License

Same as the parent project.
