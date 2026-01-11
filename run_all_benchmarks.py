"""
Run All Benchmarks Sequentially
Executes all MCP benchmarks and generates analysis
"""

import subprocess
import sys
import time
from pathlib import Path


def run_benchmark(script_name: str) -> bool:
    """Run a single benchmark script"""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            timeout=300  # 5 minute timeout per benchmark
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: {script_name} timed out")
        return False
    except Exception as e:
        print(f"ERROR: {script_name} failed with {e}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          MCP GREEN COMPUTING BENCHMARK SUITE                      ║
║                                                                   ║
║  This suite runs all benchmarks to generate empirical data        ║
║  for your survey paper on energy-efficient MCP servers.           ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Ensure we're in the benchmarks directory
    script_dir = Path(__file__).parent
    
    # Create results directory
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    benchmarks = [
        ("Token Consumption", "token_benchmark.py"),
        ("Serialization Overhead", "serialization_benchmark.py"),
        ("Caching Effectiveness", "caching_benchmark.py"),
        ("Scalability", "scalability_benchmark.py"),
        ("Energy Consumption", "energy_benchmark.py"),
    ]
    
    results = {}
    start_time = time.time()
    
    for name, script in benchmarks:
        script_path = script_dir / script
        if script_path.exists():
            success = run_benchmark(str(script_path))
            results[name] = "✓ Passed" if success else "✗ Failed"
        else:
            results[name] = "✗ Script not found"
    
    # Run analysis
    print(f"\n{'='*60}")
    print("Running Analysis")
    print('='*60)
    
    analyze_path = script_dir / "analyze_results.py"
    if analyze_path.exists():
        run_benchmark(str(analyze_path))
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"""

╔══════════════════════════════════════════════════════════════════╗
║                    BENCHMARK SUITE COMPLETE                       ║
╚══════════════════════════════════════════════════════════════════╝

Results Summary:
""")
    
    for name, status in results.items():
        print(f"  {name:.<40} {status}")
    
    print(f"""
Total Time: {total_time:.1f} seconds

Output Files:
  • results/token_benchmark_results.json
  • results/serialization_benchmark_results.json
  • results/caching_benchmark_results.json
  • results/scalability_benchmark_results.json
  • results/energy_benchmark_results.json
  • results/latex/*.tex (LaTeX tables for paper)

Next Steps:
  1. Review results in the results/ directory
  2. Copy LaTeX tables to your paper
  3. Update paper with empirical validation section
  4. Consider running benchmarks multiple times for statistical significance
""")


if __name__ == "__main__":
    main()
