"""
Experiments that need no GPU.

  E6  caching against live HTTP services, replacing the simulated service
      latencies used in the main caching benchmark
  E8  tokenizer sensitivity, and what non-English tool descriptions cost
  E9  server-side cost of progressive disclosure, and the catalogue size below
      which that orchestration outweighs the tokens it saves

    python cpu_experiments.py             # all three
    python cpu_experiments.py --only E6
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

OUT = Path("results")

# Small, unauthenticated, rate-limit-friendly endpoints. Availability is checked
# before use and recorded, so a dead endpoint is visible rather than silently
# changing the workload.
ENDPOINTS = [
    "https://httpbin.org/uuid",
    "https://api.github.com/zen",
    "https://dog.ceo/api/breeds/list/all",
    "https://catfact.ninja/fact",
    "https://worldtimeapi.org/api/timezone/Etc/UTC",
]


# ---------------------------------------------------------------------------
# E6: caching against live services
# ---------------------------------------------------------------------------

def http_get(url: str, timeout: float = 12.0) -> tuple[bytes | None, float]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mcp-bench/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return body, (time.perf_counter() - t0) * 1000.0
    except Exception:
        return None, (time.perf_counter() - t0) * 1000.0


def e6(n_requests: int = 120, seed: int = 42) -> dict:
    print("E6: caching against live HTTP services")
    live = []
    for u in ENDPOINTS:
        body, ms = http_get(u)
        if body is not None:
            live.append(u)
        print(f"  {'up  ' if body is not None else 'down'} {ms:7.1f} ms  {u}")
    if not live:
        return {"error": "no endpoint reachable", "endpoints_tried": ENDPOINTS}

    out = {"endpoints_live": live, "endpoints_tried": ENDPOINTS,
           "n_requests": n_requests, "scenarios": {}}

    for label, unique_frac in (("low repetition", 0.80), ("medium repetition", 0.50),
                               ("high repetition", 0.20), ("very high repetition", 0.05)):
        rng = random.Random(seed)
        n_unique = max(1, int(n_requests * unique_frac))
        keys = [(rng.choice(live), i % n_unique) for i in range(n_requests)]
        rng.shuffle(keys)

        # uncached: every request goes to the network
        t0 = time.perf_counter()
        lat_nc = [http_get(u)[1] for u, _ in keys]
        nc_total = (time.perf_counter() - t0) * 1000.0

        # cached: exact-match on (url, key)
        cache, hits = {}, 0
        t0 = time.perf_counter()
        lat_c = []
        for u, k in keys:
            s = time.perf_counter()
            if (u, k) in cache:
                _ = cache[(u, k)]
                hits += 1
            else:
                cache[(u, k)] = http_get(u)[0]
            lat_c.append((time.perf_counter() - s) * 1000.0)
        c_total = (time.perf_counter() - t0) * 1000.0

        out["scenarios"][label] = {
            "unique_fraction": unique_frac,
            "no_cache_total_ms": round(nc_total, 1),
            "cached_total_ms": round(c_total, 1),
            "hit_rate": hits / len(keys),
            "speedup": round(nc_total / c_total, 2) if c_total else None,
            "no_cache_p50_ms": round(statistics.median(lat_nc), 2),
            "cached_p50_ms": round(statistics.median(lat_c), 2),
            "no_cache_p95_ms": round(sorted(lat_nc)[int(0.95 * (len(lat_nc) - 1))], 2),
            "cached_p95_ms": round(sorted(lat_c)[int(0.95 * (len(lat_c) - 1))], 2),
        }
        s = out["scenarios"][label]
        print(f"  {label:22s} hit {s['hit_rate']*100:5.1f}%  "
              f"speedup {s['speedup']}x  (p50 {s['no_cache_p50_ms']} -> {s['cached_p50_ms']} ms)")
    return out


# ---------------------------------------------------------------------------
# E8: tokenizer sensitivity and non-English cost
# ---------------------------------------------------------------------------

def e8(catalogue, sample: int = 400, seed: int = 42) -> dict:
    print("E8: tokenizer sensitivity")
    rng = random.Random(seed)
    idx = rng.sample(range(len(catalogue)), min(sample, len(catalogue)))
    defs = [json.dumps({"name": catalogue[i]["name"],
                        "description": catalogue[i]["description"],
                        "parameters": catalogue[i]["parameters"]}, ensure_ascii=False)
            for i in idx]

    counters = {}
    try:
        import tiktoken
        for enc in ("cl100k_base", "o200k_base"):
            e = tiktoken.get_encoding(enc)
            counters[enc] = lambda t, e=e: len(e.encode(t))
    except Exception as ex:
        print(f"  tiktoken unavailable: {ex}")
    # Qwen2.5 and Qwen3.5 are both included because the vocabulary grew
    # sharply between them, from 151,936 to 248,320, and a larger vocabulary
    # is the one change that could move token counts on its own.
    for hf in ("Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3.5-4B",
               "BAAI/bge-small-en-v1.5"):
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(hf)
            counters[hf] = lambda t, tk=tk: len(tk.encode(t))
        except Exception as ex:
            print(f"  {hf} unavailable: {type(ex).__name__}")

    per_tok = {name: [fn(d) for d in defs] for name, fn in counters.items()}
    out = {"n_definitions": len(defs), "tokenizers": {}}
    for name, counts in per_tok.items():
        out["tokenizers"][name] = {
            "total": sum(counts),
            "mean_per_definition": round(statistics.mean(counts), 2),
        }
    if per_tok:
        base = "cl100k_base" if "cl100k_base" in per_tok else list(per_tok)[0]
        b = out["tokenizers"][base]["total"]
        for name in out["tokenizers"]:
            out["tokenizers"][name]["ratio_to_" + base] = round(
                out["tokenizers"][name]["total"] / b, 4)
        out["baseline"] = base
        spread = [v["total"] for v in out["tokenizers"].values()]
        out["max_disagreement_percent"] = round(100 * (max(spread) / min(spread) - 1), 2)
        for name, v in out["tokenizers"].items():
            print(f"  {name:34s} {v['mean_per_definition']:7.1f} tok/def  "
                  f"ratio {v.get('ratio_to_' + base)}")

    # Non-English cost. The public registry is almost entirely ASCII, which is
    # itself worth recording, so the sample is small and is reported as such.
    non_ascii = [t for t in catalogue
                 if any(ord(c) > 127 for c in (t["description"] or ""))]
    cjk = [t for t in non_ascii
           if any(0x4E00 <= ord(c) <= 0x9FFF for c in (t["description"] or ""))]
    out["multilingual"] = {
        "catalogue_size": len(catalogue),
        "non_ascii_descriptions": len(non_ascii),
        "cjk_descriptions": len(cjk),
        "ascii_share_percent": round(100 * (1 - len(non_ascii) / len(catalogue)), 3),
    }
    if cjk and counters:
        rows = {}
        for name, fn in counters.items():
            tpc = [fn(t["description"]) / max(1, len(t["description"])) for t in cjk]
            rows[name] = round(statistics.mean(tpc), 4)
        eng = rng.sample([t for t in catalogue if t not in non_ascii], min(200, len(catalogue)))
        rows_eng = {}
        for name, fn in counters.items():
            tpc = [fn(t["description"]) / max(1, len(t["description"]))
                   for t in eng if t["description"]]
            rows_eng[name] = round(statistics.mean(tpc), 4)
        out["multilingual"]["tokens_per_character_cjk"] = rows
        out["multilingual"]["tokens_per_character_english"] = rows_eng
        out["multilingual"]["note"] = (
            f"only {len(cjk)} CJK descriptions exist in the registry, so the "
            f"per-character figures are indicative rather than a population estimate")
    print(f"  registry is {out['multilingual']['ascii_share_percent']}% ASCII "
          f"({len(non_ascii)} non-ASCII descriptions)")
    return out


# ---------------------------------------------------------------------------
# E9: server-side cost of progressive disclosure
# ---------------------------------------------------------------------------

def e9(catalogue, reps: int = 40, seed: int = 42) -> dict:
    """Progressive disclosure shifts work to the server. Find where that costs
    more than it saves.

    The server pays to filter and serialise a shortlist. The client saves the
    tokens it no longer sends. Below some catalogue size the first outweighs the
    second, and a practitioner should know where that is.
    """
    print("E9: server-side cost of progressive disclosure")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ntok = lambda s: len(enc.encode(s))
    except Exception:
        ntok = lambda s: max(1, len(s) // 4)

    global _ENCODER, _CORPUS_VECS
    _ENCODER = _CORPUS_VECS = None
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as _np
        _ENCODER = SentenceTransformer("BAAI/bge-small-en-v1.5")
        _CORPUS_VECS = _np.load("data/dense_embeddings.npz")["tools"]
    except Exception as ex:
        print(f"  embedding retriever unavailable ({type(ex).__name__}); "
              f"reporting the lexical selector only")

    rng = random.Random(seed)
    out = {"reps": reps, "rows": []}
    for n in (5, 10, 25, 50, 100, 250, 500, 1000):
        pool = [catalogue[i] for i in rng.sample(range(len(catalogue)), min(n, len(catalogue)))]
        verbose_payload = json.dumps([{"name": t["name"], "description": t["description"],
                                       "parameters": t["parameters"]} for t in pool],
                                     ensure_ascii=False)
        v_tokens = ntok(verbose_payload)

        t0 = time.perf_counter()
        for _ in range(reps):
            json.dumps([{"name": t["name"], "description": t["description"],
                         "parameters": t["parameters"]} for t in pool], ensure_ascii=False)
        v_ms = (time.perf_counter() - t0) * 1000.0 / reps

        # progressive, lexical shortlist: the cheapest possible selector
        t0 = time.perf_counter()
        for _ in range(reps):
            q = "issues repository"
            scored = sorted(pool, key=lambda t: -sum(
                w in (t["description"] or "").lower() for w in q.split()))
            short = scored[:5]
            p_payload = json.dumps([{"name": t["name"], "description": t["description"],
                                     "parameters": t["parameters"]} for t in short],
                                   ensure_ascii=False)
        p_ms = (time.perf_counter() - t0) * 1000.0 / reps
        p_tokens = ntok(p_payload)

        # progressive, embedding retriever: what a deployment actually runs. The
        # query must be encoded on every request, which is the real orchestration
        # cost and is what decides whether disclosure pays for itself.
        emb_ms = None
        if _ENCODER is not None:
            import numpy as _np
            vecs = _CORPUS_VECS[:len(pool)]
            t0 = time.perf_counter()
            for _ in range(max(3, reps // 8)):
                qv = _ENCODER.encode(["issues repository"], normalize_embeddings=True,
                                     show_progress_bar=False)[0]
                order = _np.argsort(-(vecs @ qv))[:5]
                json.dumps([{"name": pool[i]["name"],
                             "description": pool[i]["description"],
                             "parameters": pool[i]["parameters"]} for i in order],
                           ensure_ascii=False)
            emb_ms = (time.perf_counter() - t0) * 1000.0 / max(3, reps // 8)

        out["rows"].append({
            "catalogue_size": n,
            "verbose_tokens": v_tokens,
            "progressive_tokens": p_tokens,
            "tokens_saved": v_tokens - p_tokens,
            "verbose_serialise_ms": round(v_ms, 4),
            "progressive_lexical_ms": round(p_ms, 4),
            "progressive_embedding_ms": round(emb_ms, 4) if emb_ms else None,
            "extra_server_ms_lexical": round(p_ms - v_ms, 4),
            "extra_server_ms_embedding": round(emb_ms - v_ms, 4) if emb_ms else None,
        })
        r = out["rows"][-1]
        print(f"  n={n:4d}  saved {r['tokens_saved']:7d} tok  "
              f"lexical {r['extra_server_ms_lexical']:+.3f} ms  "
              f"embedding {r['extra_server_ms_embedding']}")

    # Crossover, priced with a measured joules-per-input-token figure if the GPU
    # run has produced one, and a CPU power figure for the server side.
    jpt = None
    p2 = Path("results/gpu/e2_prefill_decode.json")
    if p2.exists():
        rows = [r for r in json.loads(p2.read_text(encoding="utf-8"))["rows"]
                if r.get("prefill_j_per_token")]
        if rows:
            jpt = statistics.mean(r["prefill_j_per_token"] for r in rows)
    out["prefill_j_per_token"] = jpt
    if jpt:
        cpu_w = 15.0
        for r in out["rows"]:
            extra = r.get("extra_server_ms_embedding")
            if extra is None:
                extra = r["extra_server_ms_lexical"]
            r["client_j_saved"] = round(r["tokens_saved"] * jpt, 6)
            r["server_j_extra"] = round(max(0.0, extra) / 1000.0 * cpu_w, 6)
            r["net_j_saved"] = round(r["client_j_saved"] - r["server_j_extra"], 6)
        neg = [r["catalogue_size"] for r in out["rows"] if r.get("net_j_saved", 1) <= 0]
        out["crossover_note"] = (
            f"assumed {cpu_w} W server CPU draw; disclosure is a net loss at "
            f"catalogue sizes {neg}" if neg else
            f"assumed {cpu_w} W server CPU draw; disclosure is a net gain at every "
            f"size measured")
        print(f"  {out['crossover_note']}")
    else:
        out["crossover_note"] = ("run run.py first to obtain joules per input "
                                 "token; only time is reported here")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    only = {s.strip().upper() for s in a.only.split(",") if s.strip()}

    catalogue = json.loads(Path("data/mcp_tools_catalogue.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    res = {}
    if not only or "E6" in only:
        res["e6_real_network_caching"] = e6(40 if a.quick else 120)
    if not only or "E8" in only:
        res["e8_tokenizer_sensitivity"] = e8(catalogue, 100 if a.quick else 400)
    if not only or "E9" in only:
        res["e9_disclosure_overhead"] = e9(catalogue, 10 if a.quick else 40)

    p = OUT / "cpu_experiments_results.json"
    existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    existing.update(res)
    p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
