# GPU run provenance

These are the artefacts returned from the hardware-instrumented run. Every GPU
energy figure in the paper and the supplement comes from the two files beside
this one.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 15.92 GB, driver 616.64 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| torch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Operating system | Windows |
| GPU energy counter | NVML `nvmlDeviceGetTotalEnergyConsumption`, live |
| CPU energy counter | Intel RAPL, unavailable. RAPL is exposed only on Linux |

Because RAPL is a Linux interface, this run reports GPU joules only. GPU energy
is the dominant term for local inference, so the comparisons the paper draws
from these numbers are unaffected. The CPU-side accounting the paper needs comes
from `cpu_experiments.py`, which does not require a GPU.

## Model

`Qwen/Qwen3.5-4B`, bfloat16, run as `python run.py --model qwen35`.

The model identity is verifiable from the files themselves rather than taken on
trust. E1 and E2 record input-token counts for payloads this repository builds
deterministically from `data/mcp_tools_catalogue.json`. Re-tokenising those exact
payloads reproduces 324, 1,306, 3,821 and 11,465 tokens only under the Qwen3.5
tokenizer; the Qwen2.5 tokenizer gives 322, 1,305, 3,821 and 11,463. Qwen3.5-4B
is a hybrid model: 24 of its 32 layers use linear attention and 8 use full
attention. The prefill share reported in E2 is specific to that design, which is
why the paper states the architecture wherever it quotes the split.

## Files

| File | Contents |
|---|---|
| `SUMMARY.md` | environment probe, E1 disclosure energy, E2 prefill/decode split, and the E3 to E5 scores inline |
| `followups.json` | E3 query construction, E4 MCP-Zero server stage, E5 EATS ladder with joules per query |

The per-phase JSON that `run.py` writes alongside these two files (`env.json`,
`idle_baseline.json`, `e1_disclosure_energy.json`, `e2_prefill_decode.json`,
`e3_natural_queries.json`, `e4_server_descriptions.json`) was not retained from
this run. `SUMMARY.md` carries every quantity the paper cites from E1 and E2, and
`followups.json` is the complete E3 to E5 output. Rerunning `python run.py
--model qwen35` regenerates the full set.

## Reproducing

```bash
python run.py --model qwen35      # E1 to E4, then scores E3 to E5
python gpu_followups.py results/gpu/qwen35   # rescore without rerunning the GPU
```
