# Google Colab setup (MCP Green Computing Benchmarks)

These benchmarks are pure-Python and will run on Colab.

Notes/limits:
- Performance numbers can vary a lot on Colab due to shared VMs.
- The energy benchmark uses CPU-based estimates, not true wall-power. On Colab it is **especially** noisy.

## Option A: Repo already in Colab (recommended)

In a Colab notebook cell:

```bash
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_FOLDER>/benchmarks
!python -m pip install -U pip
!python -m pip install tiktoken aiohttp matplotlib numpy msgpack psutil
!python run_all_benchmarks.py
```

## Option B: Upload this folder as a zip

1. Zip the `GreenComputingPaper` folder on your machine.
2. In Colab: upload the zip (Files sidebar).
3. Then run:

```bash
!unzip -q GreenComputingPaper.zip
%cd GreenComputingPaper/benchmarks
!python -m pip install -U pip
!python -m pip install tiktoken aiohttp matplotlib numpy msgpack psutil
!python run_all_benchmarks.py
```

## Output

After running, results are written under:
- `benchmarks/results/*.json`
- `benchmarks/results/latex/*.tex`

Download `benchmarks/results/latex/` to copy tables into your LaTeX paper.
