"""
GPU experiment runner.

One command produces every hardware-instrumented result the paper needs:

    python run.py

Two models run by default, one conventional transformer and one 2026 hybrid whose
layers are mostly linear attention, because E2's answer depends on which of the
two a deployment uses. Each model writes to results/gpu/<model>/ and a combined
comparison lands in results/gpu/SUMMARY.md.

Phases, each written out as it completes so an interruption does not lose
finished work:

  0  environment probe: GPU, driver, torch, and which energy counters are live
  E1 end-to-end energy for verbose against progressive-disclosure tool payloads,
     including a failed agent loop, measured in joules rather than modelled
  E2 prefill against decode decomposition with the KV cache on, across a
     multi-turn tool loop, so the share of energy that input-token savings can
     actually reach is measured rather than assumed
  E3 discovery under natural-language queries written by the model, instead of
     queries derived from the target tool's own description
  E4 MCP-Zero's router given a model-written server description for its first
     stage, which is what its design assumes
  E5 the EATS ablation scored in joules per query

Options:
    --quick               small sample counts, for a smoke test
    --model qwen3b,qwen35 comma-separated, run in order. Default is both.
                          qwen3b  Qwen2.5-3B-Instruct, bf16, full attention
                          qwen35  Qwen3.5-4B, bf16, hybrid linear attention
                          qwen35s Qwen3.5-2B, bf16, if 4B will not fit
                          qwen35l Qwen3.5-9B, 4-bit
                          qwen7b  Qwen2.5-7B-Instruct, 4-bit
    --skip E1,E4          skip phases
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from energy_meter import EnergyMeter, GPUMeter, RAPLMeter, environment, measure_idle

OUT = Path("results/gpu")

# Two model families are run by default, because the prefill/decode result in E2
# is architecture-dependent and a single family cannot show that.
#
#   qwen3b   Qwen2.5-3B-Instruct. A conventional transformer: every layer is full
#            attention, so the KV cache and the prefill cost both grow with the
#            prompt. This is the architecture almost all deployed assistants use.
#   qwen35   Qwen3.5-4B. A 2026 hybrid: 24 of its 32 layers use linear attention
#            and only 8 use full attention, so three quarters of the network
#            costs time linear in prompt length and holds a fixed-size state.
#            Prompt length should matter less here, and E2 measures how much.
#
# "attn" records that difference so the results carry it. "vlm" marks a model
# whose checkpoint is multimodal and therefore loads through a different class.
MODELS = {
    "qwen3b": {"name": "Qwen/Qwen2.5-3B-Instruct", "dtype": "bfloat16",
               "attn": "full", "vlm": False, "vram_gb": 6.2},
    "qwen7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "dtype": "nf4",
               "attn": "full", "vlm": False, "vram_gb": 5.6},
    "qwen35": {"name": "Qwen/Qwen3.5-4B", "dtype": "bfloat16",
               "attn": "hybrid-linear", "vlm": True, "vram_gb": 9.4},
    "qwen35s": {"name": "Qwen/Qwen3.5-2B", "dtype": "bfloat16",
                "attn": "hybrid-linear", "vlm": True, "vram_gb": 4.6},
    "qwen35l": {"name": "Qwen/Qwen3.5-9B", "dtype": "nf4",
                "attn": "hybrid-linear", "vlm": True, "vram_gb": 6.8},
}
DEFAULT_MODELS = "qwen3b,qwen35"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name: str, payload: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote {p}")


def load_model(key: str, device: str):
    import torch
    from transformers import AutoTokenizer
    spec = MODELS[key]
    name, dtype = spec["name"], spec["dtype"]
    log(f"loading {name} ({dtype}, {spec['attn']} attention); "
        f"first run downloads weights")
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"dtype": torch.bfloat16}
    if dtype == "nf4":
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")}

    # Qwen3.5 checkpoints are multimodal, so their architecture is registered
    # under the image-text class rather than the causal-language one. We feed
    # them text only; the loader just has to pick the right class.
    model = None
    if spec["vlm"]:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                name, device_map=device, **kw)
        except Exception as e:                     # older transformers, or a
            log(f"  image-text loader unavailable ({e}); trying causal LM")
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name, device_map=device, **kw)
    model = model.eval()

    # Qwen3.5 emits a reasoning block before its answer. That block is real work
    # and its energy is measured, but it must not crowd the answer out of the
    # token budget, and it must not end up inside the text we score.
    thinking = "qwen3.5" in name.lower() or "qwen3_5" in name.lower()
    return tok, model, {"thinking": thinking, "extra_new": 192 if thinking else 0}


THINK = re.compile(r"<think>.*?</think>", re.S)


def gen(model, tok, prompt: str, max_new: int, device: str, cfg=None):
    """Generate and report the token split, so energy can be attributed."""
    import torch
    cfg = cfg or {}
    ids = tok(prompt, return_tensors="pt").to(device)
    n_in = int(ids.input_ids.shape[1])
    budget = max_new + int(cfg.get("extra_new", 0))
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    n_out = int(out.shape[1]) - n_in
    txt = tok.decode(out[0, n_in:], skip_special_tokens=True)
    if cfg.get("thinking"):
        stripped = THINK.sub(" ", txt)
        # an unterminated block means the budget ran out mid-reasoning; keep
        # whatever followed the opening tag rather than returning the reasoning
        if "<think>" in stripped:
            stripped = stripped.split("<think>")[0]
        if stripped.strip():
            txt = stripped
    return txt, n_in, n_out


# ---------------------------------------------------------------------------
# payload construction, shared by E1 and E2
# ---------------------------------------------------------------------------

def build_payloads(catalogue, n_tools_verbose: int, n_tools_progressive: int):
    """Verbose exposes every tool definition; progressive exposes a shortlist."""
    verbose = json.dumps([{"name": t["name"], "description": t["description"],
                           "parameters": t["parameters"]}
                          for t in catalogue[:n_tools_verbose]], ensure_ascii=False)
    progressive = json.dumps([{"name": t["name"], "description": t["description"],
                               "parameters": t["parameters"]}
                              for t in catalogue[:n_tools_progressive]], ensure_ascii=False)
    task = ("You are an agent with access to the tools listed above. "
            "Name the single tool best suited to the request and give its "
            "arguments as JSON. Request: fetch the open issues for a repository.")
    return (f"Available tools:\n{verbose}\n\n{task}",
            f"Available tools:\n{progressive}\n\n{task}")


# ---------------------------------------------------------------------------
# E1: end-to-end energy, verbose against progressive disclosure
# ---------------------------------------------------------------------------

def e1(model, tok, device, catalogue, meters, reps: int, idle: dict):
    log("E1: end-to-end energy, verbose against progressive disclosure")
    verbose_p, prog_p = build_payloads(catalogue, 120, 8)
    conditions = {"verbose": verbose_p, "progressive": prog_p}

    # A failing agent that re-reads the full catalogue each turn is the waste
    # case the sustainability argument is really about, so it is measured too.
    conditions["verbose_failure_loop"] = "\n\n".join([verbose_p] * 3)

    results = {}
    for cond, prompt in conditions.items():
        runs = []
        for r in range(reps):
            with EnergyMeter(*meters) as m:
                _text, n_in, n_out = gen(model, tok, prompt, 64, device)
            e = m.result()
            e.update(input_tokens=n_in, output_tokens=n_out)
            for k, base in (("gpu_j", "gpu_w"), ("cpu_j", "cpu_w")):
                if e[k] is not None and idle.get(base) is not None:
                    e[k + "_net"] = e[k] - idle[base] * e["seconds"]
            runs.append(e)
        results[cond] = {
            "runs": runs,
            "input_tokens": runs[0]["input_tokens"],
            "gpu_j_mean": _mean(runs, "gpu_j"),
            "gpu_j_net_mean": _mean(runs, "gpu_j_net"),
            "cpu_j_mean": _mean(runs, "cpu_j"),
            "seconds_mean": _mean(runs, "seconds"),
        }
        log(f"  {cond:22s} in={runs[0]['input_tokens']:6d} tok  "
            f"gpu={results[cond]['gpu_j_mean']}, s={results[cond]['seconds_mean']:.2f}")

    v, p = results["verbose"], results["progressive"]
    if v["gpu_j_mean"] and p["gpu_j_mean"]:
        results["summary"] = {
            "token_reduction_percent": 100 * (1 - p["input_tokens"] / v["input_tokens"]),
            "gpu_energy_reduction_percent": 100 * (1 - p["gpu_j_mean"] / v["gpu_j_mean"]),
            "joules_per_1k_input_tokens_verbose":
                1000 * v["gpu_j_mean"] / v["input_tokens"],
        }
    return results


def _mean(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return round(statistics.mean(vals), 4) if vals else None


# ---------------------------------------------------------------------------
# E2: prefill against decode, with the KV cache on
# ---------------------------------------------------------------------------

def e2(model, tok, device, catalogue, meters, reps: int):
    """Separate the two phases and measure each.

    Input-token savings act on prefill. If a workload spends most of its energy
    in decode, cutting input tokens cannot save proportionally, which is the
    limit on the paper's central energy argument. The split is measured here
    rather than assumed in either direction.
    """
    import torch
    log("E2: prefill against decode with KV cache")
    sizes = [8, 40, 120]
    out = {"context_sizes": sizes, "rows": []}

    for n in sizes:
        prompt, _ = build_payloads(catalogue, n, 4)
        ids = tok(prompt, return_tensors="pt").to(device)
        n_in = int(ids.input_ids.shape[1])

        pre, dec = [], []
        for _ in range(reps):
            with torch.no_grad():
                with EnergyMeter(*meters) as m:
                    o = model(**ids, use_cache=True)
                    torch.cuda.synchronize() if device.startswith("cuda") else None
                pre.append(m.result())
                past = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
                with EnergyMeter(*meters) as m:
                    for _ in range(64):
                        o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                        past = o.past_key_values
                        nxt = o.logits[:, -1:].argmax(-1)
                    torch.cuda.synchronize() if device.startswith("cuda") else None
                dec.append(m.result())

        pj, dj = _mean(pre, "gpu_j"), _mean(dec, "gpu_j")
        row = {
            "tools_in_context": n,
            "input_tokens": n_in,
            "decode_tokens": 64,
            "prefill_gpu_j": pj,
            "decode_gpu_j": dj,
            "prefill_s": _mean(pre, "seconds"),
            "decode_s": _mean(dec, "seconds"),
            "prefill_share": round(pj / (pj + dj), 4) if pj and dj else None,
            "prefill_j_per_token": round(pj / n_in, 6) if pj else None,
            "decode_j_per_token": round(dj / 64, 6) if dj else None,
        }
        out["rows"].append(row)
        log(f"  {n:4d} tools, {n_in:6d} in-tok: prefill {pj} J, decode {dj} J, "
            f"prefill share {row['prefill_share']}")

    shares = [r["prefill_share"] for r in out["rows"] if r["prefill_share"]]
    if shares:
        out["prefill_share_range"] = [min(shares), max(shares)]
    return out


# ---------------------------------------------------------------------------
# E3 and E4: model-written queries and server descriptions
# ---------------------------------------------------------------------------

def write_natural_queries(model, tok, device, catalogue, idx, cfg=None, log_every=25):
    """Ask the model for a user request that the tool would satisfy.

    Queries elsewhere in this suite are derived from the target tool's own
    description, which favours lexical matching. These are written without the
    tool name and in a user's voice, so recall is measured on something closer
    to a real request.
    """
    qs = []
    for n, i in enumerate(idx):
        t = catalogue[i]
        prompt = (f"A software tool is described as: \"{t['description']}\"\n"
                  f"Write one short, natural request a user might type that this "
                  f"tool would satisfy. Do not name the tool. Reply with the "
                  f"request only.")
        txt, _, _ = gen(model, tok, prompt, 40, device, cfg)
        qs.append(" ".join(txt.strip().split())[:200])
        if n % log_every == 0:
            log(f"    query {n}/{len(idx)}")
    return qs


def write_server_descriptions(model, tok, device, servers, cfg=None, log_every=50):
    """MCP-Zero's first stage matches a model-written server description."""
    outs = []
    for n, s in enumerate(servers):
        prompt = (f"Server name: {s['name']}\nSummary: {s.get('summary','')[:300]}\n\n"
                  f"In one sentence, describe what kind of tasks this server "
                  f"provides tools for. Reply with the sentence only.")
        txt, _, _ = gen(model, tok, prompt, 40, device, cfg)
        outs.append(" ".join(txt.strip().split())[:300])
        if n % log_every == 0:
            log(f"    server {n}/{len(servers)}")
    return outs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_one(key: str, args, env, cat, servers, meters) -> int:
    """Every phase for one model, into its own results directory."""
    global OUT
    OUT = Path("results/gpu") / key
    OUT.mkdir(parents=True, exist_ok=True)
    spec = MODELS[key]
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    print("\n" + "-" * 72)
    log(f"model {key}: {spec['name']} ({spec['attn']} attention)")
    print("-" * 72)

    reps = 2 if args.quick else 5
    n_nat = 40 if args.quick else 300
    n_srv = 40 if args.quick else len(servers)
    tag = {"model": spec["name"], "model_key": key, "attention": spec["attn"]}

    save("env", {**env, "model": tag})
    tok, model, cfg = load_model(key, args.device)
    log(f"model ready (reasoning block: {cfg['thinking']})")
    try:
        return _phases(key, args, tag, tok, model, cfg, cat, servers, meters,
                       skip, reps, n_nat, n_srv)
    finally:
        # release the weights whether the phases finished or raised, or the
        # next model has no room to load
        del model, tok
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _phases(key, args, tag, tok, model, cfg, cat, servers, meters,
            skip, reps, n_nat, n_srv) -> int:

    idle = measure_idle(3.0 if args.quick else 10.0, *meters)
    save("idle_baseline", idle)
    log(f"idle: gpu {idle.get('gpu_w')} W, cpu {idle.get('cpu_w')} W")

    if "E1" not in skip:
        save("e1_disclosure_energy",
             {**tag, "reps": reps, "idle": idle,
              "conditions": e1(model, tok, args.device, cat, meters, reps, idle)})

    if "E2" not in skip:
        save("e2_prefill_decode",
             {**tag, "reps": reps,
              **e2(model, tok, args.device, cat, meters, reps)})

    if "E3" not in skip or "E5" not in skip:
        log(f"E3: writing {n_nat} natural-language queries")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cat), size=n_nat, replace=False).tolist()
        with EnergyMeter(*meters) as m:
            queries = write_natural_queries(model, tok, args.device, cat, idx, cfg)
        save("e3_natural_queries",
             {**tag, "targets": idx, "queries": queries,
              "generation_energy": m.result()})

    if "E4" not in skip:
        log(f"E4: writing {n_srv} server descriptions for MCP-Zero stage one")
        with EnergyMeter(*meters) as m:
            descs = write_server_descriptions(model, tok, args.device,
                                              servers[:n_srv], cfg)
        save("e4_server_descriptions",
             {**tag, "servers": [s["name"] for s in servers[:n_srv]],
              "descriptions": descs, "generation_energy": m.result()})

    log(f"GPU phases complete for {key}; scoring the generated artefacts")
    import subprocess
    rc = subprocess.run([sys.executable, "gpu_followups.py", str(OUT)]).returncode
    if rc != 0:
        log(f"scoring step failed; the raw outputs in {OUT} are intact and "
            f"gpu_followups.py {OUT} can be rerun on its own")
        return rc

    write_summary()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODELS,
                    help="comma-separated keys, run in order: "
                         + ", ".join(MODELS) + f" (default {DEFAULT_MODELS})")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()

    keys = [k.strip() for k in args.model.split(",") if k.strip()]
    bad = [k for k in keys if k not in MODELS]
    if bad or not keys:
        print(f"unknown model key(s): {bad or 'none given'}. "
              f"Choose from: {', '.join(MODELS)}")
        return 2

    Path("results/gpu").mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("MCP sustainability suite: GPU experiments")
    print("=" * 72)

    env = environment()
    env["args"] = vars(args)
    g = env["gpu"]
    log(f"GPU: {g.get('name')} | NVML energy: {g.get('available')} | "
        f"RAPL: {env['cpu_rapl'].get('available')}")
    if not env.get("torch", {}).get("cuda_available"):
        print("\nCUDA is not available to torch. See guide.md, section 2: an "
              "RTX 50-series card needs a cu128 build of torch.")
        return 1
    if not g.get("available"):
        log("WARNING: NVML energy counter unavailable; GPU joules will be null")
    if not env["cpu_rapl"].get("available"):
        log(f"NOTE: CPU energy unavailable ({env['cpu_rapl'].get('error')}). "
            f"GPU joules are still measured and are the dominant term.")

    cat = json.loads(Path("data/mcp_tools_catalogue.json").read_text(encoding="utf-8"))
    servers = json.loads(Path("data/mcp_servers.json").read_text(encoding="utf-8"))
    log(f"catalogue: {len(cat)} tools, {len(servers)} servers")
    log(f"models to run: {', '.join(keys)}")
    # a top-level copy so the instrument check is findable without
    # knowing which models ran
    (Path("results/gpu") / "env.json").write_text(
        json.dumps({**env, "models": keys}, indent=2), encoding="utf-8")

    meters = (GPUMeter(), RAPLMeter())
    failed = []
    for key in keys:
        try:
            if run_one(key, args, env, cat, servers, meters) != 0:
                failed.append(key)
        except Exception as e:
            # one model failing must not throw away the other model's results
            log(f"model {key} FAILED: {type(e).__name__}: {e}")
            (Path("results/gpu") / key).mkdir(parents=True, exist_ok=True)
            (Path("results/gpu") / key / "ERROR.txt").write_text(
                f"{type(e).__name__}: {e}\n", encoding="utf-8")
            failed.append(key)

    write_comparison([k for k in keys if k not in failed])
    if failed:
        log(f"finished with failures: {', '.join(failed)}. "
            f"Completed models are still in results/gpu/. Send the folder anyway.")
        return 1
    log("done. Everything is in results/gpu/. Send that folder back.")
    return 0


def write_comparison(keys):
    """Cross-model digest. The prefill share is the number that differs."""
    if not keys:
        return
    rows = ["# GPU run: cross-model comparison", "",
            "E2 asks how much of a step's energy input tokens can reach. A model",
            "whose layers are all full attention pays a superlinear prefill cost in",
            "prompt length; a hybrid whose layers are mostly linear attention does",
            "not. Both are reported so the disclosure argument is not resting on one",
            "architecture.", ""]
    rows += ["| Model | Attention | Input tokens | Prefill J | Decode J | Prefill share |",
             "|---|---|---|---|---|---|"]
    for k in keys:
        p = Path("results/gpu") / k / "e2_prefill_decode.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("rows", []):
            rows.append(f"| {d.get('model', k)} | {d.get('attention','?')} | "
                        f"{r['input_tokens']} | {r['prefill_gpu_j']} | "
                        f"{r['decode_gpu_j']} | {r['prefill_share']} |")
    rows += ["", "## E1 disclosure energy", "",
             "| Model | Tokens cut | GPU energy cut |", "|---|---|---|"]
    for k in keys:
        p = Path("results/gpu") / k / "e1_disclosure_energy.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        s_ = d.get("conditions", {}).get("summary")
        if s_:
            rows.append(f"| {d.get('model', k)} | "
                        f"{s_['token_reduction_percent']:.1f}% | "
                        f"{s_['gpu_energy_reduction_percent']:.1f}% |")
    rows += ["", "Per-model detail is in results/gpu/<model>/SUMMARY.md.", ""]
    (Path("results/gpu") / "SUMMARY.md").write_text(chr(10).join(rows),
                                                    encoding="utf-8")
    log("wrote results/gpu/SUMMARY.md")


def write_summary():
    """A short human-readable digest next to the JSON."""
    rows = ["# GPU run summary", ""]
    env = json.loads((OUT / "env.json").read_text(encoding="utf-8"))
    g = env.get("gpu", {})
    rows += [f"- GPU: {g.get('name')} ({g.get('memory_total_gb')} GB), "
             f"driver {g.get('driver')}",
             f"- torch {env.get('torch',{}).get('version')}, "
             f"CUDA {env.get('torch',{}).get('cuda_version')}, "
             f"capability {env.get('torch',{}).get('capability')}",
             f"- NVML energy counter: {g.get('available')}",
             f"- Intel RAPL: {env.get('cpu_rapl',{}).get('available')} "
             f"{env.get('cpu_rapl',{}).get('error','')}", ""]
    p = OUT / "e1_disclosure_energy.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))["conditions"]
        rows += ["## E1 disclosure energy", ""]
        for k, v in d.items():
            if k == "summary":
                continue
            rows.append(f"- {k}: {v['input_tokens']} input tokens, "
                        f"{v['gpu_j_mean']} J GPU, {v['seconds_mean']:.2f} s")
        if "summary" in d:
            s_ = d["summary"]
            rows += ["", f"- tokens cut {s_['token_reduction_percent']:.1f}%, "
                     f"GPU energy cut {s_['gpu_energy_reduction_percent']:.1f}%"]
        rows.append("")
    p = OUT / "e2_prefill_decode.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        rows += ["## E2 prefill against decode", ""]
        for r in d["rows"]:
            rows.append(f"- {r['input_tokens']} input tokens: prefill "
                        f"{r['prefill_gpu_j']} J, decode {r['decode_gpu_j']} J, "
                        f"prefill share {r['prefill_share']}")
        rows.append("")
    p = OUT / "followups.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        rows += ["## E3 to E5", "", "```", json.dumps(d, indent=2)[:3000], "```"]
    (OUT / "SUMMARY.md").write_text(chr(10).join(rows), encoding="utf-8")
    log(f"wrote {OUT / 'SUMMARY.md'}")


if __name__ == "__main__":
    sys.exit(main())
