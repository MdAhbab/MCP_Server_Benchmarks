"""
Hardware energy instrumentation.

Two counters, both read directly rather than modelled:

  GPU   NVML's total energy counter, a monotonic millijoule accumulator for the
        whole board. Present on Volta and later. This is the term that matters
        for LLM inference.
  CPU   Intel RAPL through the Linux powercap sysfs interface, in microjoules.
        Linux only. RAPL is not exposed on Windows without a signed driver, and
        WSL2 does not pass it through.

Both counters wrap, so deltas are taken modulo the counter width.

Every measurement records which instruments were live, so results state what was
measured rather than implying more than was available:

    with EnergyMeter() as m:
        run_workload()
    print(m.result())    # {'gpu_j': ..., 'cpu_j': ..., 'seconds': ...}

An idle baseline should be subtracted for attributable energy; measure_idle()
provides it.
"""

from __future__ import annotations

import glob
import os
import platform
import time

_UJ_WRAP = 1 << 63


class GPUMeter:
    """NVML board energy counter, in millijoules."""

    def __init__(self, index: int = 0):
        self.ok = False
        self.handle = None
        self.name = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            n = pynvml.nvmlDeviceGetName(self.handle)
            self.name = n.decode() if isinstance(n, bytes) else n
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.ok = True
        except Exception as e:
            self.error = str(e)

    def read_j(self) -> float | None:
        if not self.ok:
            return None
        try:
            return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0
        except Exception:
            return None

    def power_w(self) -> float | None:
        if not self.ok:
            return None
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return None

    def info(self) -> dict:
        d = {"available": self.ok, "name": self.name}
        if not self.ok:
            d["error"] = getattr(self, "error", "unknown")
            return d
        try:
            p = self._nvml
            d["driver"] = p.nvmlSystemGetDriverVersion()
            if isinstance(d["driver"], bytes):
                d["driver"] = d["driver"].decode()
            mem = p.nvmlDeviceGetMemoryInfo(self.handle)
            d["memory_total_gb"] = round(mem.total / 1024 ** 3, 2)
        except Exception:
            pass
        return d


class RAPLMeter:
    """Intel RAPL package energy through powercap sysfs, in microjoules."""

    def __init__(self):
        self.zones = []
        self.ok = False
        if platform.system() != "Linux":
            self.error = "RAPL is exposed only on Linux"
            return
        for path in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            ef = os.path.join(path, "energy_uj")
            nf = os.path.join(path, "name")
            if not os.path.exists(ef):
                continue
            try:
                with open(ef) as f:
                    f.read()
                name = open(nf).read().strip() if os.path.exists(nf) else os.path.basename(path)
                self.zones.append((name, ef))
            except PermissionError:
                self.error = (f"{ef} is not readable. Grant access with: "
                              f"sudo chmod -R a+r /sys/class/powercap/intel-rapl*")
            except Exception as e:
                self.error = str(e)
        self.ok = bool(self.zones)
        if not self.ok and not hasattr(self, "error"):
            self.error = "no intel-rapl powercap zones found"

    def read_j(self) -> float | None:
        if not self.ok:
            return None
        total = 0.0
        for _name, ef in self.zones:
            try:
                total += int(open(ef).read().strip()) / 1e6
            except Exception:
                return None
        return total

    def info(self) -> dict:
        d = {"available": self.ok, "zones": [n for n, _ in self.zones]}
        if not self.ok:
            d["error"] = getattr(self, "error", "unknown")
        return d


class EnergyMeter:
    """Context manager returning joules consumed between enter and exit."""

    def __init__(self, gpu: GPUMeter | None = None, cpu: RAPLMeter | None = None):
        self.gpu = gpu if gpu is not None else GPUMeter()
        self.cpu = cpu if cpu is not None else RAPLMeter()
        self._r = {}

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._g0 = self.gpu.read_j()
        self._c0 = self.cpu.read_j()
        return self

    def __exit__(self, *exc):
        t1 = time.perf_counter()
        g1, c1 = self.gpu.read_j(), self.cpu.read_j()
        self._r = {
            "seconds": t1 - self._t0,
            "gpu_j": _delta(self._g0, g1),
            "cpu_j": _delta(self._c0, c1),
            "gpu_measured": self.gpu.ok,
            "cpu_measured": self.cpu.ok,
        }
        return False

    def result(self) -> dict:
        return dict(self._r)


def _delta(a, b):
    if a is None or b is None:
        return None
    d = b - a
    if d < 0:                       # counter wrapped
        d += _UJ_WRAP / 1e6
    return d


def measure_idle(seconds: float = 5.0, gpu=None, cpu=None) -> dict:
    """Idle draw, so active measurements can be reported net of baseline."""
    with EnergyMeter(gpu, cpu) as m:
        time.sleep(seconds)
    r = m.result()
    out = {"seconds": r["seconds"]}
    for k in ("gpu_j", "cpu_j"):
        out[k] = r[k]
        out[k.replace("_j", "_w")] = (r[k] / r["seconds"]) if r[k] is not None else None
    return out


def environment() -> dict:
    """Everything needed to interpret the numbers in a results file."""
    g, c = GPUMeter(), RAPLMeter()
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": g.info(),
        "cpu_rapl": c.info(),
    }
    try:
        import torch
        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))
            if torch.cuda.is_available() else None,
        }
    except Exception as e:
        env["torch"] = {"error": str(e)}
    return env


if __name__ == "__main__":
    import json
    print(json.dumps(environment(), indent=2))
    print("\nidle baseline:")
    print(json.dumps(measure_idle(3.0), indent=2))
