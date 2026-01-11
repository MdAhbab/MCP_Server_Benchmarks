#!/usr/bin/env bash
set -euo pipefail

# MCP Benchmarking Environment Setup Script (macOS/Linux)
# Usage:
#   bash benchmarks/setup_environment.sh
# Optional:
#   MCP_BENCHMARK_DIR=~/MCP_Benchmarks bash benchmarks/setup_environment.sh

BENCHMARK_DIR="${MCP_BENCHMARK_DIR:-$HOME/MCP_Benchmarks}"
SUBDIRS=(servers results logs figures)

printf "=== MCP Benchmarking Environment Setup (macOS/Linux) ===\n"

printf "\nCreating directory structure...\n"
mkdir -p "$BENCHMARK_DIR"
for d in "${SUBDIRS[@]}"; do
  mkdir -p "$BENCHMARK_DIR/$d"
done
printf "Created: %s\n" "$BENCHMARK_DIR"

printf "\n=== Checking Python ===\n"
if command -v python3 >/dev/null 2>&1; then
  python3 --version
  PY=python3
elif command -v python >/dev/null 2>&1; then
  python --version
  PY=python
else
  printf "Python not found. Please install Python 3.10+ and retry.\n" >&2
  exit 1
fi

printf "\n=== Creating virtual environment ===\n"
VENV_DIR="$BENCHMARK_DIR/.venv"
"$PY" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

printf "\n=== Installing Python dependencies ===\n"
# Matches benchmarks + optional server deps used in the paper
PACKAGES=(mcp tiktoken aiohttp matplotlib numpy fastapi uvicorn msgpack psutil)
python -m pip install "${PACKAGES[@]}"

printf "\n=== Writing requirements.txt ===\n"
printf "%s\n" "${PACKAGES[@]}" > "$BENCHMARK_DIR/requirements.txt"

printf "\n=== Collecting system information ===\n"
python - << 'PY'
import json
import os
import platform

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

info = {
    "os": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python": platform.python_version(),
    "cpu_cores": os.cpu_count() or None,
}

if psutil is not None:
    try:
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass

out_dir = os.environ.get("MCP_BENCHMARK_DIR", os.path.expanduser("~/MCP_Benchmarks"))
out_path = os.path.join(out_dir, "system_info.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)

print(f"System info saved to {out_path}")
PY

printf "\n=== Setup Complete ===\n"
printf "Benchmark directory: %s\n" "$BENCHMARK_DIR"
printf "Next steps:\n"
printf "  1. Run benchmarks from this repo: (cd benchmarks && python run_all_benchmarks.py)\n"
printf "  2. Or copy benchmark scripts into %s and run them there\n" "$BENCHMARK_DIR"
