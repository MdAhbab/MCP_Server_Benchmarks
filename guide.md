# Running the GPU experiments

Thanks for lending the GPU. Everything is one command; this page covers the setup
and the one thing that reliably goes wrong on a 50-series card.

Target machine: Linux, NVIDIA RTX 5070 Ti (16 GB), 32 GB RAM.

---

## 1. Clone

```bash
git clone https://github.com/MdAhbab/MCP_Server_Benchmarks.git
cd MCP_Server_Benchmarks
```

The tool catalogue and its embeddings are committed, so there is nothing else to
download by hand. Model weights are fetched automatically on first run.

---

## 2. Install PyTorch, and please read this part

The RTX 5070 Ti is Blackwell, compute capability **12.0** (`sm_120`). A plain
`pip install torch` may install a build compiled only for older architectures.
It will import fine and then fail at the first CUDA call with something like
`no kernel image is available for execution on the device`.

Install from the CUDA 12.8 index explicitly:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

You want `True` and `(12, 0)`. If capability prints but a small matmul crashes,
the build is wrong; reinstall from the `cu128` index.

Then the rest:

```bash
pip install -r requirements.txt
```

---

## 3. Optional but worth 30 seconds: CPU energy

The script reads Intel RAPL counters for CPU energy. They are often root-only:

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl*
```

Skip it if you would rather not. GPU energy is the dominant term and is read
through NVML, which needs no permissions. The results record which counters were
live, so nothing is silently assumed.

On an AMD CPU there are no `intel-rapl` zones and this is expected; GPU energy
still works.

---

## 4. Check the environment

```bash
python energy_meter.py
```

Prints the GPU, driver, torch build, and whether each energy counter is
readable. `"available": true` under `gpu` is the one that matters.

---

## 5. Run

```bash
python run.py --quick     # ~10 minutes, confirms everything works
python run.py             # the real run
```

Do the `--quick` pass first. If it completes, the full run will too.

**Expected**

| | |
|---|---|
| Model download | Qwen2.5-3B-Instruct, about 6 GB, first run only |
| Peak VRAM | roughly 7-8 GB, comfortable on 16 GB |
| Quick run | 10-15 minutes |
| Full run | 60-90 minutes, most of it writing queries in phase E3 |

Please keep the machine otherwise idle while it runs. The script measures an
idle baseline first and subtracts it, so background load distorts the numbers.

Options if you need them:

```bash
python run.py --model qwen7b     # 7B in 4-bit, only if you want to
python run.py --skip E1,E2       # skip phases
python run.py --device cuda:1    # second GPU
```

---

## 6. Send back

Everything lands in `results/gpu/`:

```
env.json                    hardware, driver, which counters were live
idle_baseline.json          idle draw, subtracted from the measurements
e1_disclosure_energy.json   joules for verbose vs progressive tool payloads
e2_prefill_decode.json      energy split between prefill and decode
e3_natural_queries.json     queries the model wrote
e4_server_descriptions.json server descriptions the model wrote
followups.json              scored results
SUMMARY.md                  readable digest
```

Zip that folder and send it over:

```bash
zip -r gpu_results.zip results/gpu
```

---

## 7. If something breaks

**`no kernel image is available for execution on the device`**
The torch build does not include `sm_120`. Go back to section 2.

**`CUDA out of memory`**
Something else is using the card. Check with `nvidia-smi`. If it persists:
`python run.py --model qwen3b` is already the small default, so lower the work
instead with `--quick`.

**`nvmlDeviceGetTotalEnergyConsumption` fails or `pynvml` missing**
`pip install nvidia-ml-py`. Note it is `nvidia-ml-py`, not the older abandoned
`pynvml` package. The run still completes without it; GPU joules come back null.

**`NameError: Ed25519PrivateKey`** or a `cryptography` import error
`pip install cryptography`. Only the attestation demo needs it, not the GPU run.

**Hugging Face rate limit or a download stall**
`export HF_TOKEN=<your token>` and rerun. Downloads resume.

**It died halfway**
Each phase writes as it finishes, so completed phases are already on disk.
Rerun with `--skip` listing what already succeeded, for example
`python run.py --skip E1,E2`.

Any other failure: send `results/gpu/env.json` plus the console output and that
is usually enough to diagnose it.
