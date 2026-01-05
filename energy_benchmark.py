"""
Energy Consumption Benchmark for MCP Servers
Measures power consumption across different workload patterns

Note: For accurate power measurements, use hardware tools like:
- Intel Power Gadget
- HWiNFO64
- Kill-A-Watt power meter

This script provides CPU-based estimates as a baseline.
"""

import time
import json
import statistics
import threading
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


def _try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "google.colab" in sys.modules


@dataclass
class PowerMeasurement:
    """Power measurement data point"""
    timestamp: float
    cpu_percent: float
    estimated_power_watts: float


def get_cpu_utilization() -> float:
    """Get current CPU utilization percentage"""
    psutil = _try_import_psutil()
    if psutil is not None:
        # interval>0 yields a real sample rather than a cached value.
        try:
            return float(psutil.cpu_percent(interval=0.1))
        except Exception:
            pass

    try:
        if os.name == "nt":
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(result.stdout.strip())
        # Fallback: load average scaled by CPU count (best-effort)
        if hasattr(os, "getloadavg"):
            load1 = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            return max(0.0, min(100.0, (load1 / cpu_count) * 100.0))
    except Exception:
        return 0.0

    return 0.0


def get_system_info() -> Dict[str, Any]:
    """Get system information for energy calculations"""
    psutil = _try_import_psutil()

    # Cross-platform baseline
    cpu_name = platform.processor() or platform.uname().processor or "Unknown"
    cpu_cores = os.cpu_count() or 4
    cpu_max_clock_mhz: int | float = 0
    ram_gb: float = 0.0

    # Prefer psutil for memory and CPU frequency
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            ram_gb = round(vm.total / (1024**3), 2)
        except Exception:
            pass

        try:
            freq = psutil.cpu_freq()
            if freq and freq.max:
                cpu_max_clock_mhz = int(freq.max)
        except Exception:
            pass

    # Windows-specific: WMI for CPU details
    if os.name == "nt":
        try:
            cpu_result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-WmiObject -Class Win32_Processor | Select-Object Name, MaxClockSpeed, NumberOfCores | ConvertTo-Json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            cpu_info = json.loads(cpu_result.stdout)
            cpu_name = cpu_info.get("Name", cpu_name)
            cpu_cores = int(cpu_info.get("NumberOfCores", cpu_cores))
            cpu_max_clock_mhz = cpu_info.get("MaxClockSpeed", cpu_max_clock_mhz or 3000)

            if not ram_gb:
                ram_result = subprocess.run(
                    [
                        "powershell",
                        "-Command",
                        "[math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                ram_gb = float(ram_result.stdout.strip())
        except Exception:
            pass

    # macOS: sysctl memory size if psutil isn't available
    if (not ram_gb) and sys.platform == "darwin":
        try:
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            ram_gb = round(mem_bytes / (1024**3), 2)
        except Exception:
            pass

    # Linux: /proc/meminfo fallback
    if (not ram_gb) and sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / (1024**2), 2)
                        break
        except Exception:
            pass

    if not ram_gb:
        ram_gb = 16.0

    if not cpu_max_clock_mhz:
        cpu_max_clock_mhz = 3000

    return {
        "platform": platform.platform(),
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "cpu_max_clock_mhz": cpu_max_clock_mhz,
        "ram_gb": ram_gb,
        "estimated_tdp_w": estimate_tdp(cpu_name),
        "psutil_available": psutil is not None,
        "note": "Power values are CPU-based estimates; use a hardware meter for publication-quality power measurements.",
    }


def estimate_tdp(cpu_name: str) -> int:
    """Estimate TDP based on CPU name (rough approximation)"""
    cpu_lower = cpu_name.lower()

    # Apple Silicon (very rough)
    if "apple" in cpu_lower or "m1" in cpu_lower or "m2" in cpu_lower or "m3" in cpu_lower:
        return 20
    
    # Desktop CPUs
    if "i9" in cpu_lower or "ryzen 9" in cpu_lower:
        return 125
    elif "i7" in cpu_lower or "ryzen 7" in cpu_lower:
        return 95
    elif "i5" in cpu_lower or "ryzen 5" in cpu_lower:
        return 65
    elif "i3" in cpu_lower or "ryzen 3" in cpu_lower:
        return 45
    
    # Laptop CPUs (U/H series)
    if "u" in cpu_lower:
        return 15
    elif "h" in cpu_lower:
        return 45
    
    # Default
    return 65


class PowerMonitor:
    """Background power monitoring"""
    
    def __init__(self, tdp_watts: int = 65, sample_interval_ms: int = 500):
        self.tdp = tdp_watts
        self.interval = sample_interval_ms / 1000
        self.measurements: List[PowerMeasurement] = []
        self._running = False
        self._thread = None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            cpu_pct = get_cpu_utilization()
            # Estimate power: P = TDP * (CPU% / 100) * 0.7 + base_power
            # 0.7 factor accounts for non-linear power scaling
            # Base power ~30% TDP for idle
            estimated_power = self.tdp * 0.3 + self.tdp * 0.7 * (cpu_pct / 100)
            
            self.measurements.append(PowerMeasurement(
                timestamp=time.time(),
                cpu_percent=cpu_pct,
                estimated_power_watts=estimated_power
            ))
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        self.measurements = []
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()
    
    def stop(self) -> List[PowerMeasurement]:
        """Stop monitoring and return measurements"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self.measurements
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics from measurements"""
        if not self.measurements:
            return {}
        
        powers = [m.estimated_power_watts for m in self.measurements]
        cpus = [m.cpu_percent for m in self.measurements]
        duration = self.measurements[-1].timestamp - self.measurements[0].timestamp
        
        avg_power = statistics.mean(powers)
        energy_j = avg_power * duration
        energy_wh = energy_j / 3600
        
        return {
            "duration_seconds": duration,
            "samples": len(self.measurements),
            "avg_cpu_percent": statistics.mean(cpus),
            "max_cpu_percent": max(cpus),
            "avg_power_watts": avg_power,
            "peak_power_watts": max(powers),
            "min_power_watts": min(powers),
            "energy_joules": energy_j,
            "energy_wh": energy_wh
        }


# Workload simulations
def idle_workload(duration_seconds: int = 10):
    """Baseline idle workload"""
    time.sleep(duration_seconds)


def json_processing_workload(duration_seconds: int = 10):
    """Simulate MCP JSON processing"""
    end_time = time.time() + duration_seconds
    
    # Sample MCP payloads
    tool_definitions = [
        {
            "name": f"tool_{i}",
            "description": f"Description for tool {i} " * 5,
            "inputSchema": {"type": "object", "properties": {f"p{j}": {"type": "string"} for j in range(5)}}
        }
        for i in range(50)
    ]
    
    while time.time() < end_time:
        # Serialize
        json_str = json.dumps(tool_definitions)
        # Deserialize
        json.loads(json_str)


def heavy_computation_workload(duration_seconds: int = 10):
    """Simulate heavy computation (solver-like)"""
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        # CPU-intensive operations
        result = 0
        for i in range(100000):
            result += i * i
        # Small sleep to prevent 100% CPU lock
        time.sleep(0.001)


def mixed_mcp_workload(duration_seconds: int = 10):
    """Simulate realistic MCP workload mix"""
    end_time = time.time() + duration_seconds
    
    tool_defs = [{"name": f"tool_{i}", "desc": "x" * 100} for i in range(20)]
    
    while time.time() < end_time:
        # 60% JSON processing
        for _ in range(6):
            json.dumps(tool_defs)
            json.loads(json.dumps(tool_defs))
        
        # 30% Light computation
        for _ in range(3):
            sum(range(10000))
        
        # 10% Idle (network wait simulation)
        time.sleep(0.01)


def run_benchmark():
    """Execute the complete energy benchmark"""
    
    print("=" * 70)
    print("MCP Energy Consumption Benchmark")
    print("=" * 70)
    
    # Get system info
    sys_info = get_system_info()
    
    print(f"\nSystem Information:")
    if sys_info.get("platform"):
        print(f"  Platform: {sys_info['platform']}")
    print(f"  CPU: {sys_info['cpu_name']}")
    print(f"  Cores: {sys_info['cpu_cores']}")
    print(f"  Estimated TDP: {sys_info['estimated_tdp_w']}W")
    print(f"  RAM: {sys_info['ram_gb']}GB")

    if _is_colab():
        print("\nNote: Running in Google Colab. CPU/power estimates may be noisy due to shared infrastructure.")
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": sys_info,
        "workloads": []
    }
    
    tdp = sys_info['estimated_tdp_w']
    workload_duration = 15  # seconds per workload
    
    workloads = [
        ("Idle (Baseline)", idle_workload),
        ("JSON Processing (MCP typical)", json_processing_workload),
        ("Heavy Computation (Solver)", heavy_computation_workload),
        ("Mixed MCP Workload", mixed_mcp_workload),
    ]
    
    print(f"\n--- Running Workload Benchmarks ({workload_duration}s each) ---\n")
    
    for name, workload_func in workloads:
        print(f"Running: {name}...")
        
        monitor = PowerMonitor(tdp_watts=tdp, sample_interval_ms=500)
        monitor.start()
        
        workload_func(workload_duration)
        
        monitor.stop()
        stats = monitor.get_stats()
        
        print(f"  Avg CPU: {stats['avg_cpu_percent']:.1f}%")
        print(f"  Avg Power: {stats['avg_power_watts']:.1f}W")
        print(f"  Energy: {stats['energy_wh']:.4f}Wh")
        
        results["workloads"].append({
            "name": name,
            "duration_seconds": workload_duration,
            **stats
        })
        
        # Cool-down between workloads
        time.sleep(2)
    
    # Comparative Analysis
    print("\n" + "=" * 70)
    print("Comparative Analysis")
    print("=" * 70 + "\n")
    
    idle = results["workloads"][0]
    json_proc = results["workloads"][1]
    heavy = results["workloads"][2]
    mixed = results["workloads"][3]
    
    print(f"{'Workload':<30} {'Avg Power (W)':<15} {'vs Idle':<15} {'Energy (Wh)':<15}")
    print("-" * 75)
    
    for w in results["workloads"]:
        overhead = ((w['avg_power_watts'] - idle['avg_power_watts']) / idle['avg_power_watts'] * 100)
        overhead_str = f"+{overhead:.1f}%" if overhead > 0 else f"{overhead:.1f}%"
        print(f"{w['name']:<30} {w['avg_power_watts']:<15.1f} {overhead_str:<15} {w['energy_wh']:<15.4f}")
    
    # Scale to production workloads
    print("\n--- Scaled Energy Estimates ---\n")
    
    # Assume 1000 requests/minute for 8 hours
    requests_per_day = 1000 * 60 * 8
    
    # Estimate time per request from JSON processing rate
    requests_per_second = 100  # Conservative estimate
    active_time_seconds = requests_per_day / requests_per_second
    
    print(f"Production scenario: {requests_per_day:,} MCP requests/day")
    print(f"Estimated active processing time: {active_time_seconds/3600:.2f} hours")
    
    # Energy calculations
    json_energy_per_hour = json_proc['energy_wh'] * (3600 / workload_duration)
    daily_energy = json_energy_per_hour * (active_time_seconds / 3600)
    
    print(f"\nEnergy consumption estimates:")
    print(f"  Per hour (active): {json_energy_per_hour:.2f}Wh")
    print(f"  Daily (production): {daily_energy:.2f}Wh")
    print(f"  Monthly: {daily_energy * 30 / 1000:.2f}kWh")
    print(f"  Yearly: {daily_energy * 365 / 1000:.2f}kWh")
    
    # Cost estimation (average US electricity rate)
    kwh_cost = 0.12  # $0.12 per kWh
    yearly_cost = (daily_energy * 365 / 1000) * kwh_cost
    print(f"\nEstimated yearly electricity cost: ${yearly_cost:.2f}")
    
    results["production_estimates"] = {
        "requests_per_day": requests_per_day,
        "daily_energy_wh": daily_energy,
        "monthly_energy_kwh": daily_energy * 30 / 1000,
        "yearly_energy_kwh": daily_energy * 365 / 1000,
        "yearly_cost_usd": yearly_cost
    }
    
    # Optimization impact
    print("\n--- Optimization Impact Estimates ---\n")
    
    # From token benchmark: 98.7% reduction possible
    optimized_energy = daily_energy * 0.013  # 98.7% reduction
    
    print(f"Current approach (verbose): {daily_energy:.2f}Wh/day")
    print(f"Optimized (progressive disclosure): {optimized_energy:.2f}Wh/day")
    print(f"Potential savings: {daily_energy - optimized_energy:.2f}Wh/day ({(1 - 0.013) * 100:.1f}%)")
    print(f"Annual savings: {(daily_energy - optimized_energy) * 365 / 1000:.2f}kWh")
    print(f"Cost savings: ${((daily_energy - optimized_energy) * 365 / 1000) * kwh_cost:.2f}/year")
    
    results["optimization_impact"] = {
        "baseline_daily_wh": daily_energy,
        "optimized_daily_wh": optimized_energy,
        "savings_percent": 98.7,
        "annual_savings_kwh": (daily_energy - optimized_energy) * 365 / 1000
    }
    
    # CO2 impact
    print("\n--- Environmental Impact ---\n")
    
    # Average US grid: 0.42 kg CO2 per kWh
    co2_per_kwh = 0.42
    
    baseline_co2 = (daily_energy * 365 / 1000) * co2_per_kwh
    optimized_co2 = (optimized_energy * 365 / 1000) * co2_per_kwh
    
    print(f"Annual CO2 emissions:")
    print(f"  Baseline: {baseline_co2:.2f}kg CO2")
    print(f"  Optimized: {optimized_co2:.4f}kg CO2")
    print(f"  Reduction: {baseline_co2 - optimized_co2:.2f}kg CO2")
    
    results["environmental_impact"] = {
        "co2_per_kwh_kg": co2_per_kwh,
        "baseline_annual_co2_kg": baseline_co2,
        "optimized_annual_co2_kg": optimized_co2,
        "co2_savings_kg": baseline_co2 - optimized_co2
    }
    
    # Save results
    output_file = Path("results/energy_benchmark_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Paper summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    
    print(f"""
Energy consumption analysis for MCP server workloads:

1. Workload Characterization:
   - Idle baseline: {idle['avg_power_watts']:.1f}W
   - JSON processing (MCP typical): {json_proc['avg_power_watts']:.1f}W (+{((json_proc['avg_power_watts']-idle['avg_power_watts'])/idle['avg_power_watts']*100):.1f}% vs idle)
   - Heavy computation (solvers): {heavy['avg_power_watts']:.1f}W (+{((heavy['avg_power_watts']-idle['avg_power_watts'])/idle['avg_power_watts']*100):.1f}% vs idle)

2. Production Estimates (480K requests/day):
   - Daily energy: {daily_energy:.2f}Wh
   - Annual energy: {daily_energy * 365 / 1000:.2f}kWh
   - Annual CO2: {baseline_co2:.2f}kg

3. Green Computing Impact:
   - Progressive disclosure optimization: 98.7% token reduction
   - Potential energy savings: {(daily_energy - optimized_energy) * 365 / 1000:.2f}kWh/year
   - CO2 reduction: {baseline_co2 - optimized_co2:.2f}kg/year

Note: These are CPU-based estimates. For publication-quality results,
use hardware power meters (Intel Power Gadget, Kill-A-Watt).
""")
    
    return results


if __name__ == "__main__":
    run_benchmark()
