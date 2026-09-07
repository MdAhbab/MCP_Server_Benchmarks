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
python run.py --quick     # ~25 minutes, confirms everything works for both models
python run.py             # the real run
```

Do the `--quick` pass first. If it completes, the full run will too.

**Two models run, one after the other.** This is deliberate, not a mistake:

- **Qwen2.5-3B-Instruct**, a conventional transformer, every layer full attention
- **Qwen3.5-4B**, a 2026 hybrid, 24 of its 32 layers linear attention

The experiment that separates prefill from decode has a different answer on each,
because prompt length costs a full-attention model much more than it costs a
hybrid. Running only one would report an architecture-specific number as if it
were general. They are loaded one at a time and the first is freed before the
second loads, so only one set of weights is ever resident.

**Expected**

| | |
|---|---|
| Model download | about 15 GB total, first run only: 6 GB for Qwen2.5-3B, 9 GB for Qwen3.5-4B |
| Peak VRAM | roughly 11-12 GB, on the Qwen3.5-4B pass. Comfortable on 16 GB |
| Quick run | 20-30 minutes for both models |
| Full run | 2-3 hours for both, most of it writing queries in phase E3 |

If you are short on time or disk, `python run.py --model qwen3b` runs only the
first model, in about half the time. Please run both if you can, since the
comparison is the point.

Please keep the machine otherwise idle while it runs. The script measures an
idle baseline first and subtracts it, so background load distorts the numbers.

Options if you need them:

```bash
python run.py --model qwen3b            # the conventional model alone
python run.py --model qwen35            # the hybrid alone
python run.py --model qwen3b,qwen35s    # substitute the 2B hybrid if 4B will not fit
python run.py --model qwen35l           # Qwen3.5-9B in 4-bit, if you want a bigger one
python run.py --skip E1,E2              # skip phases
python run.py --device cuda:1           # second GPU
```

---

## 6. Send back

Everything lands in `results/gpu/`, one subdirectory per model:

```
env.json                      hardware, driver, which counters were live
SUMMARY.md                    the cross-model comparison, read this first
qwen3b/                       the conventional transformer
  env.json                    the same probe, plus which model this was
  idle_baseline.json          idle draw, subtracted from the measurements
  e1_disclosure_energy.json   joules for verbose vs progressive tool payloads
  e2_prefill_decode.json      energy split between prefill and decode
  e3_natural_queries.json     queries the model wrote
  e4_server_descriptions.json server descriptions the model wrote
  followups.json              scored results
  SUMMARY.md                  readable digest for this model
qwen35/                       the hybrid, same files
```

If one model fails, the other's results are kept and an `ERROR.txt` is written
in the failed model's folder. Send the folder either way.

Zip that folder and send it over:

```bash
zip -r gpu_results.zip results/gpu
```

---

## 7. If something breaks

**`no kernel image is available for execution on the device`**
The torch build does not include `sm_120`. Go back to section 2.

**`CUDA out of memory`**
Check `nvidia-smi` first, since anything else on the card matters at these sizes.
The hybrid pass is the heavier of the two at roughly 11-12 GB, so if only that
one fails, swap in the smaller hybrid and keep the comparison:
`python run.py --model qwen3b,qwen35s`. `--quick` also lowers the work.

**`nvmlDeviceGetTotalEnergyConsumption` fails or `pynvml` missing**
`pip install nvidia-ml-py`. Note it is `nvidia-ml-py`, not the older abandoned
`pynvml` package. The run still completes without it; GPU joules come back null.

**`NameError: Ed25519PrivateKey`** or a `cryptography` import error
`pip install cryptography`. Only the attestation demo needs it, not the GPU run.

**Hugging Face rate limit or a download stall**
`export HF_TOKEN=<your token>` and rerun. Downloads resume.

**`Unrecognized configuration class` or `qwen3_5` not known**
The installed `transformers` predates Qwen3.5. `pip install -U "transformers>=5.16"`.
Everything else in the suite works on older versions; only this model needs it.

**It died halfway**
Each phase writes as it finishes, so completed phases are already on disk.
Rerun with `--skip` listing what already succeeded, for example
`python run.py --skip E1,E2`.

Any other failure: send `results/gpu/env.json` plus the console output and that
is usually enough to diagnose it.
