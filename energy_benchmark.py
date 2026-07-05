"""
Energy Consumption Benchmark for MCP Servers (v2, revised after peer review)
Estimates power draw across workload patterns using a utilisation-based CPU
power proxy with PROCESS-LEVEL ATTRIBUTION.

Methodology
-----------
- System-wide AND benchmark-process CPU utilisation are sampled every 500 ms
  via psutil. The first sample of each window is discarded (counter warm-up).
- Estimated whole-system power uses an affine utilisation model:
      P_system = TDP * (0.30 + 0.70 * system_cpu / 100)
- Workload-ATTRIBUTED power isolates the benchmark process from background
  OS activity (this fixes the v1 anomaly where a contaminated idle baseline
  appeared to consume more power than active workloads):
      P_attributed = TDP * 0.70 * (process_cpu / n_cores) / 100
- Each workload runs for 30 s with a 3 s cool-down between workloads.
  Mean, median, and standard deviation are reported for every quantity.

IMPORTANT: These are model-based ESTIMATES, not physical wall-power
measurements. For publication-grade absolute numbers use a hardware power
meter (Kill-A-Watt), RAPL counters (Linux: perf / powertop), or Intel Power
Gadget. The relative comparison between workloads is the meaningful result.
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

WORKLOAD_DURATION_S = 30      # seconds per workload window
COOLDOWN_S = 3                # cool-down between workloads
SAMPLE_INTERVAL_MS = 500      # power-model sampling interval
WARMUP_SAMPLES_DISCARDED = 1  # first cpu_percent() call returns 0.0

# Affine utilisation power model coefficients
IDLE_POWER_FRACTION = 0.30    # base (uncore/idle) power as fraction of TDP
DYNAMIC_POWER_FRACTION = 0.70 # utilisation-proportional fraction of TDP


def _try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _is_colab() -> bool:
    return "COLAB_GPU" in os.environ or "google.colab" in sys.modules


@dataclass
class PowerSample:
    """One sampling instant of the utilisation power model."""
    timestamp: float
    system_cpu_percent: float
    process_cpu_percent: float        # normalised: 100 == all cores busy
    system_power_watts: float         # estimate incl. background processes
    attributed_power_watts: float     # estimate for the benchmark process only


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
        "python_version": platform.python_version(),
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "cpu_max_clock_mhz": cpu_max_clock_mhz,
        "ram_gb": ram_gb,
        "estimated_tdp_w": estimate_tdp(cpu_name),
        "psutil_available": psutil is not None,
        "note": (
            "Power values are utilisation-model estimates, not wall-power "
            "measurements; use a hardware meter or RAPL for absolute numbers."
        ),
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
    """Background sampler for the utilisation power model.

    Samples both system-wide CPU and the benchmark process's own CPU so that
    workload power can be attributed independently of background OS activity.
    Requires psutil.
    """

    def __init__(self, tdp_watts: int = 65, sample_interval_ms: int = SAMPLE_INTERVAL_MS):
        psutil = _try_import_psutil()
        if psutil is None:
            raise RuntimeError(
                "psutil is required for the energy benchmark (pip install psutil)"
            )
        self._psutil = psutil
        self._process = psutil.Process()
        self._cores = os.cpu_count() or 1
        self.tdp = tdp_watts
        self.interval = sample_interval_ms / 1000
        self.samples: List[PowerSample] = []
        self._running = False
        self._thread = None

    def _sample_once(self) -> PowerSample:
        sys_cpu = float(self._psutil.cpu_percent(interval=None))
        # Process.cpu_percent() can exceed 100 (sum over cores); normalise so
        # 100 means "all cores saturated by this process".
        proc_cpu = float(self._process.cpu_percent(interval=None)) / self._cores
        proc_cpu = max(0.0, min(100.0, proc_cpu))

        system_power = self.tdp * (
            IDLE_POWER_FRACTION + DYNAMIC_POWER_FRACTION * sys_cpu / 100
        )
        attributed_power = self.tdp * DYNAMIC_POWER_FRACTION * proc_cpu / 100

        return PowerSample(
            timestamp=time.time(),
            system_cpu_percent=sys_cpu,
            process_cpu_percent=proc_cpu,
            system_power_watts=system_power,
            attributed_power_watts=attributed_power,
        )

    def _monitor_loop(self):
        # Prime the stateful counters; the first reading is meaningless.
        self._psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        while self._running:
            time.sleep(self.interval)
            self.samples.append(self._sample_once())

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[PowerSample]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics over the sampling window (warm-up discarded)."""
        samples = self.samples[WARMUP_SAMPLES_DISCARDED:]
        if len(samples) < 2:
            return {}

        sys_powers = [s.system_power_watts for s in samples]
        attr_powers = [s.attributed_power_watts for s in samples]
        sys_cpus = [s.system_cpu_percent for s in samples]
        proc_cpus = [s.process_cpu_percent for s in samples]
        duration = samples[-1].timestamp - samples[0].timestamp

        avg_sys_power = statistics.mean(sys_powers)
        avg_attr_power = statistics.mean(attr_powers)

        return {
            "duration_seconds": duration,
            "samples": len(samples),
            "sample_interval_ms": self.interval * 1000,
            # CPU utilisation
            "avg_cpu_percent": statistics.mean(sys_cpus),        # system-wide
            "max_cpu_percent": max(sys_cpus),
            "std_cpu_percent": statistics.stdev(sys_cpus),
            "avg_process_cpu_percent": statistics.mean(proc_cpus),
            "std_process_cpu_percent": statistics.stdev(proc_cpus),
            # Whole-system power estimate (includes background processes)
            "avg_power_watts": avg_sys_power,
            "median_power_watts": statistics.median(sys_powers),
            "std_power_watts": statistics.stdev(sys_powers),
            "peak_power_watts": max(sys_powers),
            "min_power_watts": min(sys_powers),
            "energy_joules": avg_sys_power * duration,
            "energy_wh": avg_sys_power * duration / 3600,
            # Workload-attributed power estimate (benchmark process only)
            "avg_attributed_power_watts": avg_attr_power,
            "median_attributed_power_watts": statistics.median(attr_powers),
            "std_attributed_power_watts": statistics.stdev(attr_powers),
            "attributed_energy_joules": avg_attr_power * duration,
            "attributed_energy_wh": avg_attr_power * duration / 3600,
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
    """Simulate heavy computation (solver-like): CPU-bound, single-threaded."""
    end_time = time.time() + duration_seconds

    result = 0
    while time.time() < end_time:
        for i in range(100000):
            result += i * i


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
    print("MCP Energy Consumption Benchmark (v2: process-attributed estimates)")
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
        "methodology": {
            "power_model": (
                "P_system = TDP*(0.30 + 0.70*system_cpu/100); "
                "P_attributed = TDP*0.70*(process_cpu/n_cores)/100"
            ),
            "sample_interval_ms": SAMPLE_INTERVAL_MS,
            "workload_duration_s": WORKLOAD_DURATION_S,
            "cooldown_s": COOLDOWN_S,
            "warmup_samples_discarded": WARMUP_SAMPLES_DISCARDED,
            "isolation": (
                "Background processes are NOT terminated; instead, per-process "
                "CPU attribution isolates the benchmark workload from "
                "background activity. System-wide figures are reported "
                "separately and include background noise."
            ),
            "limitations": (
                "Utilisation-proxy model; no DVFS/frequency states, fan or "
                "thermal effects, GPU, DRAM or I/O power. Absolute values are "
                "estimates; relative workload comparisons are the meaningful "
                "quantity. Validate absolute numbers with RAPL or a wall meter."
            ),
        },
        "workloads": []
    }

    tdp = sys_info['estimated_tdp_w']
    workload_duration = WORKLOAD_DURATION_S

    workloads = [
        ("Idle (Baseline)", idle_workload),
        ("JSON Processing (MCP typical)", json_processing_workload),
        ("Heavy Computation (Solver)", heavy_computation_workload),
        ("Mixed MCP Workload", mixed_mcp_workload),
    ]

    print(f"\n--- Running Workload Benchmarks ({workload_duration}s each) ---\n")

    for name, workload_func in workloads:
        print(f"Running: {name}...")

        monitor = PowerMonitor(tdp_watts=tdp, sample_interval_ms=SAMPLE_INTERVAL_MS)
        monitor.start()

        workload_func(workload_duration)

        monitor.stop()
        stats = monitor.get_stats()

        print(f"  System CPU: {stats['avg_cpu_percent']:.1f}% (±{stats['std_cpu_percent']:.1f})")
        print(f"  Process CPU: {stats['avg_process_cpu_percent']:.1f}% (±{stats['std_process_cpu_percent']:.1f})")
        print(f"  System power est.: {stats['avg_power_watts']:.1f}W (±{stats['std_power_watts']:.1f})")
        print(f"  Attributed power est.: {stats['avg_attributed_power_watts']:.2f}W (±{stats['std_attributed_power_watts']:.2f})")
        print(f"  Attributed energy: {stats['attributed_energy_wh']:.4f}Wh")

        results["workloads"].append({
            "name": name,
            **stats
        })

        # Cool-down between workloads
        time.sleep(COOLDOWN_S)

    # Comparative Analysis
    print("\n" + "=" * 70)
    print("Comparative Analysis (workload-attributed power, background excluded)")
    print("=" * 70 + "\n")

    idle = results["workloads"][0]
    json_proc = results["workloads"][1]
    heavy = results["workloads"][2]
    mixed = results["workloads"][3]

    print(f"{'Workload':<30} {'Attr. Power (W)':<18} {'vs Idle (W)':<15} {'Energy (Wh)':<15}")
    print("-" * 78)

    for w in results["workloads"]:
        marginal = w['avg_attributed_power_watts'] - idle['avg_attributed_power_watts']
        marginal_str = "baseline" if w is idle else f"+{marginal:.2f}"
        print(f"{w['name']:<30} {w['avg_attributed_power_watts']:<18.2f} {marginal_str:<15} {w['attributed_energy_wh']:<15.4f}")

    print(
        "\nNote: whole-system estimates (incl. background processes) are in the "
        "JSON output;\nthey fluctuate with OS activity and must not be compared "
        "across workloads directly."
    )

    # Scale to production workloads
    print("\n--- Scaled Energy Estimates (model-based extrapolation) ---\n")

    # Assume 1000 requests/minute for 8 hours
    requests_per_day = 1000 * 60 * 8

    # Estimate time per request from JSON processing rate
    requests_per_second = 100  # Conservative estimate
    active_time_seconds = requests_per_day / requests_per_second

    print(f"Production scenario: {requests_per_day:,} MCP requests/day")
    print(f"Estimated active processing time: {active_time_seconds/3600:.2f} hours")

    # Energy calculations based on ATTRIBUTED power of the JSON workload
    json_energy_per_hour = json_proc['attributed_energy_wh'] * (3600 / workload_duration)
    daily_energy = json_energy_per_hour * (active_time_seconds / 3600)

    print(f"\nEnergy consumption estimates (server-side CPU only):")
    print(f"  Per hour (active): {json_energy_per_hour:.2f}Wh")
    print(f"  Daily (production): {daily_energy:.2f}Wh")
    print(f"  Monthly: {daily_energy * 30 / 1000:.2f}kWh")
    print(f"  Yearly: {daily_energy * 365 / 1000:.2f}kWh")

    # Cost estimation (average US electricity rate)
    kwh_cost = 0.12  # $0.12 per kWh
    yearly_cost = (daily_energy * 365 / 1000) * kwh_cost
    print(f"\nEstimated yearly electricity cost: ${yearly_cost:.2f}")

    results["production_estimates"] = {
        "assumptions": {
            "requests_per_day": requests_per_day,
            "requests_per_second_throughput": requests_per_second,
            "basis": "attributed (process-level) energy of the JSON workload",
            "scope": "server-side CPU only; excludes LLM inference energy",
        },
        "requests_per_day": requests_per_day,
        "daily_energy_wh": daily_energy,
        "monthly_energy_kwh": daily_energy * 30 / 1000,
        "yearly_energy_kwh": daily_energy * 365 / 1000,
        "yearly_cost_usd": yearly_cost
    }

    # Optimization impact
    print("\n--- Optimization Impact (scenario analysis, NOT a measurement) ---\n")

    # Scenario parameter: the 98.7% payload/token reduction reported by the
    # Anthropic code-execution case study and consistent with our token
    # benchmark (94.7-99.9% depending on dataset size). Applied here to the
    # server-side JSON processing volume as a what-if scenario.
    token_reduction_fraction = 0.987
    optimized_energy = daily_energy * (1 - token_reduction_fraction)

    print(f"Scenario parameter: {token_reduction_fraction*100:.1f}% payload reduction (progressive disclosure)")
    print(f"Current approach (verbose): {daily_energy:.2f}Wh/day")
    print(f"Optimized (progressive disclosure): {optimized_energy:.2f}Wh/day")
    print(f"Potential savings: {daily_energy - optimized_energy:.2f}Wh/day ({token_reduction_fraction * 100:.1f}%)")
    print(f"Annual savings: {(daily_energy - optimized_energy) * 365 / 1000:.2f}kWh")
    print(f"Cost savings: ${((daily_energy - optimized_energy) * 365 / 1000) * kwh_cost:.2f}/year")

    results["optimization_impact"] = {
        "type": "scenario_analysis",
        "scenario_parameter": (
            "98.7% payload reduction from progressive disclosure "
            "(literature-reported; our token benchmark measured 94.7-99.9%)"
        ),
        "baseline_daily_wh": daily_energy,
        "optimized_daily_wh": optimized_energy,
        "savings_percent": token_reduction_fraction * 100,
        "annual_savings_kwh": (daily_energy - optimized_energy) * 365 / 1000
    }

    # CO2 impact
    print("\n--- Environmental Impact (model-based estimate) ---\n")

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
Energy consumption analysis for MCP server workloads (utilisation-model
estimates with process-level attribution):

1. Workload Characterization (attributed power, marginal vs idle):
   - Idle baseline: {idle['avg_attributed_power_watts']:.2f}W attributed
   - JSON processing (MCP typical): {json_proc['avg_attributed_power_watts']:.2f}W (+{json_proc['avg_attributed_power_watts']-idle['avg_attributed_power_watts']:.2f}W vs idle)
   - Heavy computation (solvers): {heavy['avg_attributed_power_watts']:.2f}W (+{heavy['avg_attributed_power_watts']-idle['avg_attributed_power_watts']:.2f}W vs idle)
   - Mixed MCP workload: {mixed['avg_attributed_power_watts']:.2f}W (+{mixed['avg_attributed_power_watts']-idle['avg_attributed_power_watts']:.2f}W vs idle)

2. Production Estimates ({requests_per_day:,} requests/day, server-side CPU only):
   - Daily energy: {daily_energy:.2f}Wh
   - Annual energy: {daily_energy * 365 / 1000:.2f}kWh
   - Annual CO2: {baseline_co2:.2f}kg

3. Green Computing Impact (scenario analysis):
   - Progressive disclosure optimization: 98.7% payload reduction (scenario)
   - Potential energy savings: {(daily_energy - optimized_energy) * 365 / 1000:.2f}kWh/year
   - CO2 reduction: {baseline_co2 - optimized_co2:.2f}kg/year

Note: These are utilisation-model estimates, not wall-power measurements.
Use hardware power meters or RAPL counters for absolute validation.
""")

    return results


if __name__ == "__main__":
    run_benchmark()
