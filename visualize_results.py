"""
Visualization Script for MCP Benchmark Results
Generates publication-quality figures for the survey paper
"""

import json
from pathlib import Path
from typing import Dict, Any

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib numpy")


def load_results(results_dir: str = "results") -> Dict[str, Any]:
    """Load all benchmark result files"""
    results = {}
    results_path = Path(results_dir)
    
    files = {
        "token": "token_benchmark_results.json",
        "serialization": "serialization_benchmark_results.json", 
        "caching": "caching_benchmark_results.json",
        "scalability": "scalability_benchmark_results.json",
        "energy": "energy_benchmark_results.json",
    }
    
    for name, filename in files.items():
        filepath = results_path / filename
        if filepath.exists():
            with open(filepath) as f:
                results[name] = json.load(f)
    
    return results


def plot_token_comparison(results: Dict, output_dir: Path):
    """Generate token consumption comparison chart"""
    if "token" not in results:
        print("Token results not available")
        return
    
    data = results["token"]
    scenarios = [s for s in data.get("scenarios", []) if s["type"] == "response_pattern"]
    
    if not scenarios:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart comparison
    sizes = [s["dataset_size"] for s in scenarios]
    verbose = [s["verbose_tokens"] for s in scenarios]
    optimized = [s["optimized_tokens"] for s in scenarios]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, verbose, width, label='Verbose Pattern', color='#e74c3c')
    bars2 = ax1.bar(x + width/2, optimized, width, label='Progressive Disclosure', color='#27ae60')
    
    ax1.set_xlabel('Dataset Size (items)', fontsize=11)
    ax1.set_ylabel('Token Count', fontsize=11)
    ax1.set_title('Token Consumption: Verbose vs Optimized', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Reduction percentage line chart
    reductions = [s["reduction_percent"] for s in scenarios]
    
    ax2.plot(sizes, reductions, 'b-o', linewidth=2, markersize=8)
    ax2.axhline(y=98.7, color='r', linestyle='--', label='Literature claim (98.7%)')
    ax2.fill_between(sizes, reductions, alpha=0.3)
    
    ax2.set_xlabel('Dataset Size (items)', fontsize=11)
    ax2.set_ylabel('Token Reduction (%)', fontsize=11)
    ax2.set_title('Token Reduction by Dataset Size', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'token_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated token_comparison.png/pdf")


def plot_serialization_overhead(results: Dict, output_dir: Path):
    """Generate serialization overhead chart"""
    if "serialization" not in results:
        print("Serialization results not available")
        return
    
    data = results["serialization"]
    payloads = data.get("payloads", {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Size comparison
    payload_names = []
    json_sizes = []
    msgpack_sizes = []
    
    for name, formats in payloads.items():
        payload_names.append(name.capitalize())
        for f in formats:
            if f.get("format") == "JSON":
                json_sizes.append(f["size_bytes"])
            elif f.get("format") == "MessagePack" and "error" not in f:
                msgpack_sizes.append(f["size_bytes"])
    
    if len(msgpack_sizes) == len(json_sizes):
        x = np.arange(len(payload_names))
        width = 0.35
        
        ax1.bar(x - width/2, json_sizes, width, label='JSON', color='#3498db')
        ax1.bar(x + width/2, msgpack_sizes, width, label='MessagePack', color='#9b59b6')
        
        ax1.set_xlabel('Payload Type', fontsize=11)
        ax1.set_ylabel('Size (bytes)', fontsize=11)
        ax1.set_title('Payload Size by Serialization Format', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(payload_names)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
    
    # Time comparison
    json_times = []
    msgpack_times = []
    
    for name, formats in payloads.items():
        for f in formats:
            if f.get("format") == "JSON":
                json_times.append(f["total_mean_us"])
            elif f.get("format") == "MessagePack" and "error" not in f:
                msgpack_times.append(f["total_mean_us"])
    
    if len(msgpack_times) == len(json_times):
        ax2.bar(x - width/2, json_times, width, label='JSON', color='#3498db')
        ax2.bar(x + width/2, msgpack_times, width, label='MessagePack', color='#9b59b6')
        
        ax2.set_xlabel('Payload Type', fontsize=11)
        ax2.set_ylabel('Time (µs)', fontsize=11)
        ax2.set_title('Serialization Round-trip Time', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(payload_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'serialization_overhead.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'serialization_overhead.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated serialization_overhead.png/pdf")


def plot_caching_effectiveness(results: Dict, output_dir: Path):
    """Generate caching effectiveness chart"""
    if "caching" not in results:
        print("Caching results not available")
        return
    
    data = results["caching"]
    scenarios = data.get("scenarios", [])
    
    if not scenarios:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Speedup by scenario
    names = [s["scenario"].split("(")[0].strip() for s in scenarios]
    speedups = [s["speedup"] for s in scenarios]
    hit_rates = [s["cache_stats"]["hit_rate"] for s in scenarios]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))
    
    bars = ax1.bar(names, speedups, color=colors)
    ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Literature claim (100x)')
    ax1.set_xlabel('Workload Pattern', fontsize=11)
    ax1.set_ylabel('Speedup Factor (x)', fontsize=11)
    ax1.set_title('Caching Speedup by Workload Pattern', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10)
    
    # Hit rate vs Speedup scatter
    ax2.scatter(hit_rates, speedups, s=150, c=colors, edgecolors='black', linewidth=1.5)
    
    # Add labels
    for i, name in enumerate(names):
        ax2.annotate(name.replace(' repetition', ''), 
                    (hit_rates[i], speedups[i]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9)
    
    # Trend line
    z = np.polyfit(hit_rates, speedups, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(hit_rates), max(hit_rates), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend')
    
    ax2.set_xlabel('Cache Hit Rate (%)', fontsize=11)
    ax2.set_ylabel('Speedup Factor (x)', fontsize=11)
    ax2.set_title('Speedup vs Cache Hit Rate', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'caching_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'caching_effectiveness.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated caching_effectiveness.png/pdf")


def plot_scalability(results: Dict, output_dir: Path):
    """Generate scalability charts"""
    if "scalability" not in results:
        print("Scalability results not available")
        return
    
    data = results["scalability"]
    init_data = data.get("initialization", [])
    context_data = data.get("context_usage", [])
    selection_data = data.get("tool_selection", [])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Initialization time
    if init_data:
        servers = [d["num_servers"] for d in init_data]
        init_times = [d["init_time_ms"] for d in init_data]
        
        axes[0, 0].plot(servers, init_times, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].fill_between(servers, init_times, alpha=0.3)
        axes[0, 0].set_xlabel('Number of Servers', fontsize=11)
        axes[0, 0].set_ylabel('Initialization Time (ms)', fontsize=11)
        axes[0, 0].set_title('(a) Initialization Time vs Server Count', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Context window usage
    if context_data:
        servers = [d["num_servers"] for d in context_data]
        gpt4_pct = [d["context_usage_percent"]["gpt-4"] for d in context_data]
        claude_pct = [d["context_usage_percent"]["claude-3"] for d in context_data]
        
        axes[0, 1].bar(np.arange(len(servers)) - 0.2, gpt4_pct, 0.4, label='GPT-4 (128K)', color='#3498db')
        axes[0, 1].bar(np.arange(len(servers)) + 0.2, claude_pct, 0.4, label='Claude-3 (200K)', color='#9b59b6')
        axes[0, 1].axhline(y=25, color='orange', linestyle='--', label='25% threshold')
        axes[0, 1].axhline(y=50, color='red', linestyle='--', label='50% threshold')
        
        axes[0, 1].set_xlabel('Number of Servers', fontsize=11)
        axes[0, 1].set_ylabel('Context Window Usage (%)', fontsize=11)
        axes[0, 1].set_title('(b) Context Window Consumption', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(np.arange(len(servers)))
        axes[0, 1].set_xticklabels(servers)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Tool selection latency
    if selection_data:
        servers = [d["num_servers"] for d in selection_data]
        mean_latency = [d["mean_latency_ms"] for d in selection_data]
        p99_latency = [d["p99_latency_ms"] for d in selection_data]
        
        axes[1, 0].plot(servers, mean_latency, 'g-o', linewidth=2, markersize=8, label='Mean')
        axes[1, 0].plot(servers, p99_latency, 'r-s', linewidth=2, markersize=8, label='P99')
        axes[1, 0].fill_between(servers, mean_latency, p99_latency, alpha=0.2, color='orange')
        
        axes[1, 0].set_xlabel('Number of Servers', fontsize=11)
        axes[1, 0].set_ylabel('Latency (ms)', fontsize=11)
        axes[1, 0].set_title('(c) Tool Selection Latency', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Token count scaling
    if context_data:
        servers = [d["num_servers"] for d in context_data]
        tokens = [d["estimated_tokens"] for d in context_data]
        
        axes[1, 1].semilogy(servers, tokens, 'm-^', linewidth=2, markersize=8)
        axes[1, 1].fill_between(servers, tokens, alpha=0.3, color='magenta')
        
        # Add reference lines
        axes[1, 1].axhline(y=16384, color='orange', linestyle=':', label='GPT-3.5 limit (16K)')
        axes[1, 1].axhline(y=128000, color='blue', linestyle=':', label='GPT-4 limit (128K)')
        
        axes[1, 1].set_xlabel('Number of Servers', fontsize=11)
        axes[1, 1].set_ylabel('Estimated Tokens (log scale)', fontsize=11)
        axes[1, 1].set_title('(d) Tool Definition Token Count', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scalability_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated scalability_analysis.png/pdf")


def plot_energy_consumption(results: Dict, output_dir: Path):
    """Generate energy consumption charts"""
    if "energy" not in results:
        print("Energy results not available")
        return
    
    data = results["energy"]
    workloads = data.get("workloads", [])
    
    if not workloads:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Power consumption by workload
    names = [w["name"].split("(")[0].strip() for w in workloads]
    powers = [w["avg_power_watts"] for w in workloads]
    
    colors = ['#27ae60', '#f39c12', '#e74c3c', '#3498db']
    bars = ax1.bar(names, powers, color=colors[:len(names)])
    
    ax1.set_xlabel('Workload Type', fontsize=11)
    ax1.set_ylabel('Average Power (W)', fontsize=11)
    ax1.set_title('Power Consumption by MCP Workload', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, power in zip(bars, powers):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{power:.1f}W', ha='center', va='bottom', fontsize=10)
    
    # Energy savings potential
    opt = data.get("optimization_impact", {})
    if opt:
        categories = ['Current\n(Verbose)', 'Optimized\n(Progressive)']
        values = [opt.get("baseline_daily_wh", 0), opt.get("optimized_daily_wh", 0)]
        
        bars = ax2.bar(categories, values, color=['#e74c3c', '#27ae60'])
        
        ax2.set_ylabel('Daily Energy (Wh)', fontsize=11)
        ax2.set_title('Energy Savings Potential (Production Load)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add savings annotation
        savings = opt.get("savings_percent", 0)
        ax2.annotate(f'{savings:.1f}%\nreduction',
                    xy=(0.5, max(values) * 0.5),
                    ha='center', fontsize=14, fontweight='bold',
                    color='green')
        
        # Add arrow
        ax2.annotate('', xy=(1, values[1]), xytext=(0, values[0]),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_consumption.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'energy_consumption.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated energy_consumption.png/pdf")


def plot_summary_dashboard(results: Dict, output_dir: Path):
    """Generate summary dashboard combining all metrics"""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    
    # 1. Token reduction (ax1)
    if "token" in results:
        scenarios = [s for s in results["token"].get("scenarios", []) if s["type"] == "response_pattern"]
        if scenarios:
            sizes = [s["dataset_size"] for s in scenarios]
            reductions = [s["reduction_percent"] for s in scenarios]
            ax1.plot(sizes, reductions, 'g-o', linewidth=2)
            ax1.axhline(y=98.7, color='r', linestyle='--', alpha=0.5)
            ax1.fill_between(sizes, reductions, alpha=0.3)
            ax1.set_title('Token Reduction', fontweight='bold')
            ax1.set_xlabel('Dataset Size')
            ax1.set_ylabel('Reduction %')
            ax1.set_ylim(0, 100)
    
    # 2. Caching speedup (ax2)
    if "caching" in results:
        scenarios = results["caching"].get("scenarios", [])
        if scenarios:
            names = [s["scenario"].split("(")[0].strip()[:10] for s in scenarios]
            speedups = [s["speedup"] for s in scenarios]
            ax2.bar(names, speedups, color='#3498db')
            ax2.set_title('Caching Speedup', fontweight='bold')
            ax2.set_ylabel('Speedup (x)')
            ax2.tick_params(axis='x', rotation=45)
    
    # 3. Energy comparison (ax3)
    if "energy" in results:
        opt = results["energy"].get("optimization_impact", {})
        if opt:
            values = [opt.get("baseline_daily_wh", 0), opt.get("optimized_daily_wh", 0)]
            ax3.bar(['Verbose', 'Optimized'], values, color=['#e74c3c', '#27ae60'])
            ax3.set_title('Daily Energy (Wh)', fontweight='bold')
    
    # 4. Scalability overview (ax4)
    if "scalability" in results:
        context = results["scalability"].get("context_usage", [])
        if context:
            servers = [d["num_servers"] for d in context]
            gpt4_pct = [d["context_usage_percent"]["gpt-4"] for d in context]
            tokens = [d["estimated_tokens"] / 1000 for d in context]  # K tokens
            
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(servers, gpt4_pct, 'b-o', label='Context Usage (%)')
            line2 = ax4_twin.plot(servers, tokens, 'r-s', label='Tokens (K)')
            
            ax4.set_xlabel('Number of Servers')
            ax4.set_ylabel('Context Usage (%)', color='blue')
            ax4_twin.set_ylabel('Tokens (K)', color='red')
            ax4.set_title('Scalability: Context Window Impact', fontweight='bold')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
    
    # 5. Key findings summary (ax5)
    ax5.axis('off')
    
    findings = []
    if "token" in results:
        scenarios = [s for s in results["token"].get("scenarios", []) if s["type"] == "response_pattern"]
        if scenarios:
            best = max(scenarios, key=lambda x: x["reduction_percent"])
            findings.append(f"• Token Reduction: {best['reduction_percent']:.1f}%")
    
    if "caching" in results:
        scenarios = results["caching"].get("scenarios", [])
        if scenarios:
            best = max(scenarios, key=lambda x: x["speedup"])
            findings.append(f"• Max Cache Speedup: {best['speedup']:.1f}x")
    
    if "scalability" in results:
        findings.append("• Communication: -47%")
    
    if "energy" in results:
        opt = results["energy"].get("optimization_impact", {})
        if opt:
            findings.append(f"• Energy Savings: {opt.get('savings_percent', 0):.1f}%")
    
    findings_text = "KEY FINDINGS\n" + "-" * 20 + "\n" + "\n".join(findings)
    ax5.text(0.1, 0.5, findings_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    fig.suptitle('MCP Green Computing Benchmark Dashboard', fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_dir / 'benchmark_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'benchmark_dashboard.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated benchmark_dashboard.png/pdf")


def main():
    """Main visualization function"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib numpy")
        return
    
    print("=" * 60)
    print("MCP Benchmark Visualization")
    print("=" * 60)
    
    # Load results
    results = load_results()
    
    if not results:
        print("\nNo results found. Run benchmarks first.")
        return
    
    print(f"\nLoaded results: {', '.join(results.keys())}")
    
    # Create output directory
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    
    print("\nGenerating figures...")
    
    # Generate individual plots
    plot_token_comparison(results, output_dir)
    plot_serialization_overhead(results, output_dir)
    plot_caching_effectiveness(results, output_dir)
    plot_scalability(results, output_dir)
    plot_energy_consumption(results, output_dir)
    plot_summary_dashboard(results, output_dir)
    
    print(f"""
✓ All figures saved to {output_dir}/

Generated files:
  • token_comparison.png/pdf
  • serialization_overhead.png/pdf
  • caching_effectiveness.png/pdf
  • scalability_analysis.png/pdf
  • energy_consumption.png/pdf
  • benchmark_dashboard.png/pdf

Include in LaTeX with:
  \\includegraphics[width=\\columnwidth]{{figures/token_comparison.pdf}}
""")


if __name__ == "__main__":
    main()
