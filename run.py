"""
GPU experiment runner.

One command produces every hardware-instrumented result the paper needs:

    python run.py

Phases, each written to results/gpu/ as it completes so an interruption does not
lose finished work:

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
    --quick          small sample counts, for a smoke test
    --model NAME     qwen3b (default) or qwen7b
    --skip E1,E4     skip phases
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from energy_meter import EnergyMeter, GPUMeter, RAPLMeter, environment, measure_idle

OUT = Path("results/gpu")

MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", "bfloat16"),
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "nf4"),
}


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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name, dtype = MODELS[key]
    log(f"loading {name} ({dtype}); first run downloads weights")
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"dtype": torch.bfloat16}
    if dtype == "nf4":
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")}
    model = AutoModelForCausalLM.from_pretrained(name, device_map=device, **kw).eval()
    return tok, model


def gen(model, tok, prompt: str, max_new: int, device: str):
    """Generate and report the token split, so energy can be attributed."""
    import torch
    ids = tok(prompt, return_tensors="pt").to(device)
    n_in = int(ids.input_ids.shape[1])
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, use_cache=True)
    n_out = int(out.shape[1]) - n_in
    return tok.decode(out[0, n_in:], skip_special_tokens=True), n_in, n_out


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

def write_natural_queries(model, tok, device, catalogue, idx, log_every=25):
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
        txt, _, _ = gen(model, tok, prompt, 40, device)
        qs.append(" ".join(txt.strip().split())[:200])
        if n % log_every == 0:
            log(f"    query {n}/{len(idx)}")
    return qs


def write_server_descriptions(model, tok, device, servers, log_every=50):
    """MCP-Zero's first stage matches a model-written server description."""
    outs = []
    for n, s in enumerate(servers):
        prompt = (f"Server name: {s['name']}\nSummary: {s.get('summary','')[:300]}\n\n"
                  f"In one sentence, describe what kind of tasks this server "
                  f"provides tools for. Reply with the sentence only.")
        txt, _, _ = gen(model, tok, prompt, 40, device)
        outs.append(" ".join(txt.strip().split())[:300])
        if n % log_every == 0:
            log(f"    server {n}/{len(servers)}")
    return outs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="qwen3b", choices=list(MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("MCP sustainability suite: GPU experiments")
    print("=" * 72)

    env = environment()
    env["args"] = vars(args)
    save("env", env)
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

    meters = (GPUMeter(), RAPLMeter())
    reps = 2 if args.quick else 5
    n_nat = 40 if args.quick else 300
    n_srv = 40 if args.quick else len(servers)

    tok, model = load_model(args.model, args.device)
    log("model ready")

    idle = measure_idle(3.0 if args.quick else 10.0, *meters)
    save("idle_baseline", idle)
    log(f"idle: gpu {idle.get('gpu_w')} W, cpu {idle.get('cpu_w')} W")

    if "E1" not in skip:
        save("e1_disclosure_energy",
             {"model": MODELS[args.model][0], "reps": reps, "idle": idle,
              "conditions": e1(model, tok, args.device, cat, meters, reps, idle)})

    if "E2" not in skip:
        save("e2_prefill_decode",
             {"model": MODELS[args.model][0], "reps": reps,
              **e2(model, tok, args.device, cat, meters, reps)})

    if "E3" not in skip or "E5" not in skip:
        log(f"E3: writing {n_nat} natural-language queries")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cat), size=n_nat, replace=False).tolist()
        with EnergyMeter(*meters) as m:
            queries = write_natural_queries(model, tok, args.device, cat, idx)
        save("e3_natural_queries",
             {"model": MODELS[args.model][0], "targets": idx, "queries": queries,
              "generation_energy": m.result()})

    if "E4" not in skip:
        log(f"E4: writing {n_srv} server descriptions for MCP-Zero stage one")
        with EnergyMeter(*meters) as m:
            descs = write_server_descriptions(model, tok, args.device, servers[:n_srv])
        save("e4_server_descriptions",
             {"model": MODELS[args.model][0],
              "servers": [s["name"] for s in servers[:n_srv]],
              "descriptions": descs, "generation_energy": m.result()})

    log("GPU phases complete; scoring the generated artefacts")
    import subprocess
    rc = subprocess.run([sys.executable, "gpu_followups.py"]).returncode
    if rc != 0:
        log("scoring step failed; the raw GPU outputs in results/gpu/ are intact "
            "and gpu_followups.py can be rerun on its own")
        return rc

    write_summary()
    log("done. Everything is in results/gpu/. Send that folder back.")
    return 0


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
