#!/usr/bin/env python3
"""
Visualization script for tensor operation benchmarks.

Reads benchmark_results.csv and generates comprehensive matplotlib visualizations:
- Performance comparison between CPU and GPU for different operations
- Throughput analysis across different tensor sizes
- Scaling behavior for different data types
- Memory transfer overhead analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data(csv_path='benchmark_results.csv'):
    """Load benchmark data from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} benchmark results")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nOperations: {df['operation'].unique()}")
    print(f"Backends: {df['backend'].unique()}")
    print(f"Data types: {df['data_type'].unique()}")
    print(f"Shapes: {df['shape'].unique()}")
    
    return df


def plot_cpu_vs_gpu_operations(df, output_dir='plots'):
    """Compare CPU vs GPU performance for different operations."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter out memory transfer operations for fair comparison
    compute_ops = df[~df['operation'].str.contains('copy')].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CPU vs GPU Performance Comparison', fontsize=16, fontweight='bold')
    
    operations = ['add', 'subtract', 'multiply']
    data_types = sorted(compute_ops['data_type'].unique())
    n_types = len(data_types)
    
    # Plot 1: Duration comparison by operation
    ax = axes[0, 0]
    
    # Collect all data first
    cpu_data_all = []
    gpu_data_all = []
    operation_names = None
    
    for dtype in data_types:
        data = compute_ops[compute_ops['data_type'] == dtype]
        cpu_data = data[data['backend'] == 'cpu'].groupby('operation')['duration_ms'].mean()
        gpu_data = data[data['backend'] == 'cuda'].groupby('operation')['duration_ms'].mean()
        cpu_data_all.append(cpu_data.values)
        gpu_data_all.append(gpu_data.values)
        if operation_names is None:
            operation_names = cpu_data.index
    
    # Set up bar positions
    n_ops = len(operation_names) # type: ignore
    x = np.arange(n_ops)
    width = 0.8 / (n_types * 2)  # Total width divided by (num_types * 2 backends)
    
    # Plot bars with proper grouping
    for i, dtype in enumerate(data_types):
        offset_cpu = i * 2 * width - (n_types * width)
        offset_gpu = (i * 2 + 1) * width - (n_types * width)
        
        ax.bar(x + offset_cpu, cpu_data_all[i], width, 
               label=f'CPU ({dtype})', alpha=0.8)
        ax.bar(x + offset_gpu, gpu_data_all[i], width,
               label=f'GPU ({dtype})', alpha=0.8, hatch='//')
    
    ax.set_xlabel('Operation')
    ax.set_ylabel('Duration (ms)')
    ax.set_title('Average Duration by Operation')
    ax.set_xticks(x)
    ax.set_xticklabels(operation_names)
    # add rotation to x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(fontsize=8, ncol=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Throughput comparison
    ax = axes[0, 1]
    
    # Collect all data first
    cpu_throughput_all = []
    gpu_throughput_all = []
    
    for dtype in data_types:
        data = compute_ops[compute_ops['data_type'] == dtype]
        cpu_throughput = data[data['backend'] == 'cpu'].groupby('operation')['throughput_gb_s'].mean()
        gpu_throughput = data[data['backend'] == 'cuda'].groupby('operation')['throughput_gb_s'].mean()
        cpu_throughput_all.append(cpu_throughput.values)
        gpu_throughput_all.append(gpu_throughput.values)
    
    # Set up bar positions
    x = np.arange(n_ops)
    width = 0.8 / (n_types * 2)
    
    # Plot bars with proper grouping
    for i, dtype in enumerate(data_types):
        offset_cpu = i * 2 * width - (n_types * width)
        offset_gpu = (i * 2 + 1) * width - (n_types * width)
        
        ax.bar(x + offset_cpu, cpu_throughput_all[i], width,
               label=f'CPU ({dtype})', alpha=0.8)
        ax.bar(x + offset_gpu, gpu_throughput_all[i], width,
               label=f'GPU ({dtype})', alpha=0.8, hatch='//')
    
    ax.set_xlabel('Operation')
    ax.set_ylabel('Throughput (GB/s)')
    ax.set_title('Average Throughput by Operation')
    ax.set_xticks(x)
    ax.set_xticklabels(operation_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speedup factor (GPU vs CPU)
    ax = axes[1, 0]
    speedups = []
    labels = []
    
    for dtype in data_types:
        for op in operations:
            cpu_time = compute_ops[(compute_ops['backend'] == 'cpu') & 
                                  (compute_ops['operation'] == op) &
                                  (compute_ops['data_type'] == dtype)]['duration_ms'].mean()
            gpu_time = compute_ops[(compute_ops['backend'] == 'cuda') & 
                                  (compute_ops['operation'] == op) &
                                  (compute_ops['data_type'] == dtype)]['duration_ms'].mean()
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                speedups.append(speedup)
                labels.append(f'{op}\n({dtype})')
    
    from matplotlib import cm
    colors = cm.RdYlGn(np.linspace(0.3, 0.9, len(speedups)))  # type: ignore
    bars = ax.bar(range(len(speedups)), speedups, color=colors, alpha=0.8)
    ax.axhline(y=1, color='r', linestyle='--', label='No speedup', linewidth=2)
    ax.set_xlabel('Operation (Data Type)')
    ax.set_ylabel('Speedup Factor (CPU time / GPU time)')
    ax.set_title('GPU Speedup vs CPU')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Size vs Duration
    ax = axes[1, 1]
    for backend in ['cpu', 'cuda']:
        for dtype in data_types:
            data = compute_ops[(compute_ops['backend'] == backend) & 
                             (compute_ops['data_type'] == dtype)]
            grouped = data.groupby('size')['duration_ms'].mean()
            
            marker = 'o' if backend == 'cpu' else '^'
            linestyle = '-' if backend == 'cpu' else '--'
            ax.plot(grouped.index, grouped.values, 
                   marker=marker, linestyle=linestyle,
                   label=f'{backend.upper()} ({dtype})', alpha=0.7)
    
    ax.set_xlabel('Tensor Size (elements)')
    ax.set_ylabel('Duration (ms)')
    ax.set_title('Scaling: Duration vs Tensor Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cpu_vs_gpu_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/cpu_vs_gpu_comparison.png")
    plt.close()


def plot_memory_transfer(df, output_dir='plots'):
    """Analyze memory transfer performance."""
    Path(output_dir).mkdir(exist_ok=True)
    
    transfer_ops = df[df['operation'].str.contains('copy')].copy()
    
    if transfer_ops.empty:
        print("No memory transfer data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Memory Transfer Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Transfer time by direction
    ax = axes[0]
    for dtype in transfer_ops['data_type'].unique():
        data = transfer_ops[transfer_ops['data_type'] == dtype]
        
        for op in data['operation'].unique():
            op_data = data[data['operation'] == op].sort_values('size')
            label = f'{op} ({dtype})'
            marker = 'o' if 'cpu_to_gpu' in op else 's'
            ax.plot(op_data['size'], op_data['duration_ms'], 
                   marker=marker, label=label, alpha=0.7)
    
    ax.set_xlabel('Tensor Size (elements)')
    ax.set_ylabel('Transfer Time (ms)')
    ax.set_title('Memory Transfer Duration')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Transfer bandwidth
    ax = axes[1]
    for dtype in transfer_ops['data_type'].unique():
        data = transfer_ops[transfer_ops['data_type'] == dtype]
        
        for op in data['operation'].unique():
            op_data = data[data['operation'] == op].sort_values('size')
            label = f'{op} ({dtype})'
            marker = 'o' if 'cpu_to_gpu' in op else 's'
            ax.plot(op_data['size'], op_data['throughput_gb_s'], 
                   marker=marker, label=label, alpha=0.7)
    
    ax.set_xlabel('Tensor Size (elements)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Memory Transfer Bandwidth')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_transfer.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/memory_transfer.png")
    plt.close()


def plot_scaling_analysis(df, output_dir='plots'):
    """Analyze how performance scales with tensor size."""
    Path(output_dir).mkdir(exist_ok=True)
    
    compute_ops = df[~df['operation'].str.contains('copy')].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scaling Analysis: Performance vs Tensor Size', 
                 fontsize=16, fontweight='bold')
    
    operations = compute_ops['operation'].unique()
    
    for idx, op in enumerate(operations[:6]):  # Limit to 6 plots
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        op_data = compute_ops[compute_ops['operation'] == op]
        
        for backend in ['cpu', 'cuda']:
            for dtype in op_data['data_type'].unique():
                data = op_data[(op_data['backend'] == backend) & 
                              (op_data['data_type'] == dtype)]
                grouped = data.groupby('size').agg({
                    'duration_ms': 'mean',
                    'throughput_gb_s': 'mean'
                }).reset_index()
                
                marker = 'o' if backend == 'cpu' else '^'
                linestyle = '-' if backend == 'cpu' else '--'
                label = f'{backend.upper()} ({dtype})'
                
                ax.plot(grouped['size'], grouped['throughput_gb_s'],
                       marker=marker, linestyle=linestyle, 
                       label=label, alpha=0.7)
        
        ax.set_xlabel('Tensor Size (elements)')
        ax.set_ylabel('Throughput (GB/s)')
        ax.set_title(f'Operation: {op}')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/scaling_analysis.png")
    plt.close()


def plot_data_type_comparison(df, output_dir='plots'):
    """Compare performance across different data types."""
    Path(output_dir).mkdir(exist_ok=True)
    
    compute_ops = df[~df['operation'].str.contains('copy')].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Data Type Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Average throughput by data type
    ax = axes[0]
    throughput_by_type = compute_ops.groupby(['data_type', 'backend'])['throughput_gb_s'].mean().unstack()
    throughput_by_type.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Average Throughput (GB/s)')
    ax.set_title('Average Throughput by Data Type')
    ax.legend(title='Backend')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    
    # Plot 2: Performance ratio (throughput relative to i32)
    ax = axes[1]
    x = None
    types = []
    
    for backend in ['cpu', 'cuda']:
        ratios = []
        backend_types = []
        
        backend_data = compute_ops[compute_ops['backend'] == backend]
        i32_throughput = backend_data[backend_data['data_type'] == 'i32']['throughput_gb_s'].mean()
        
        for dtype in backend_data['data_type'].unique():
            dtype_throughput = backend_data[backend_data['data_type'] == dtype]['throughput_gb_s'].mean()
            if i32_throughput > 0:
                ratio = dtype_throughput / i32_throughput
                ratios.append(ratio)
                backend_types.append(dtype)
        
        if not types:
            types = backend_types
        
        x = np.arange(len(backend_types))
        width = 0.35
        offset = -width/2 if backend == 'cpu' else width/2
        
        bars = ax.bar(x + offset, ratios, width, label=backend.upper(), alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    if x is not None and types:
        ax.axhline(y=1, color='r', linestyle='--', label='Baseline (i32)', linewidth=2)
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Relative Throughput (vs i32)')
        ax.set_title('Throughput Relative to i32')
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_type_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/data_type_comparison.png")
    plt.close()


def plot_heatmap(df, output_dir='plots'):
    """Create heatmap of performance metrics."""
    Path(output_dir).mkdir(exist_ok=True)
    
    compute_ops = df[~df['operation'].str.contains('copy')].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Performance Heatmaps', fontsize=16, fontweight='bold')
    
    # Heatmap 1: CPU throughput
    cpu_data = compute_ops[compute_ops['backend'] == 'cpu']
    pivot_cpu = cpu_data.pivot_table(
        values='throughput_gb_s',
        index='operation',
        columns='data_type',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_cpu, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'Throughput (GB/s)'})
    axes[0].set_title('CPU Throughput')
    axes[0].set_xlabel('Data Type')
    axes[0].set_ylabel('Operation')
    
    # Heatmap 2: GPU throughput
    gpu_data = compute_ops[compute_ops['backend'] == 'cuda']
    pivot_gpu = gpu_data.pivot_table(
        values='throughput_gb_s',
        index='operation',
        columns='data_type',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_gpu, annot=True, fmt='.2f', cmap='YlGnBu',
                ax=axes[1], cbar_kws={'label': 'Throughput (GB/s)'})
    axes[1].set_title('GPU Throughput')
    axes[1].set_xlabel('Data Type')
    axes[1].set_ylabel('Operation')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/performance_heatmap.png")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("Loading benchmark data...")
    df = load_data()
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_cpu_vs_gpu_operations(df)
    plot_memory_transfer(df)
    plot_scaling_analysis(df)
    plot_data_type_comparison(df)
    plot_heatmap(df)
    
    print("\n✓ All visualizations generated successfully!")
    print("Check the 'plots/' directory for output files.")


if __name__ == '__main__':
    main()
