#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_strong_scaling(df, output='results/plots/strong_scaling.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    threads = df['threads'].values
    times = df['time'].values
    
    t1 = times[0]
    speedup = t1 / times
    efficiency = speedup / threads * 100
    
    ax1.plot(threads, speedup, 'o-', label='Measured', linewidth=2, markersize=10, color='steelblue')
    ax1.plot(threads, threads, 'k--', label='Ideal', linewidth=2, alpha=0.5)
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Strong Scaling: Speedup', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(threads)
    
    ax2.plot(threads, efficiency, 's-', label='Measured', linewidth=2, markersize=10, color='darkgreen')
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Ideal')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('Strong Scaling: Efficiency', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(threads)
    ax2.set_ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_scaling.py <scaling_data.csv>")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    print(f"Loaded {len(df)} data points")
    print(df)
    
    os.makedirs('results/plots', exist_ok=True)
    plot_strong_scaling(df)
    print("\nPlot generated successfully!")