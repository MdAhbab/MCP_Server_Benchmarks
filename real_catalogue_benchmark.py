"""
Graph-Connected Hierarchical Discovery on a real MCP tool catalogue.

The main GHD benchmark runs on a synthetic catalogue. This script repeats the
discovery experiments on the MCP-tools dataset released with MCP-Zero (Fei,
Zheng and Feng, arXiv:2506.01056): 308 servers and 2,797 tools taken from the
official Model Context Protocol repository.

Run mcp_zero_catalogue.py first to produce data/mcp_tools_catalogue.json and
data/mcp_tools_embeddings.npy, then:

    python real_catalogue_benchmark.py

What is real and what is not
----------------------------
Real: every tool name, description, parameter schema, and server assignment.
Real: the functional-equivalence groups, which are found from the released
      description embeddings, so they reflect genuine duplication across servers.
Synthetic: latency and energy values. No public MCP registry publishes
      per-tool operational metadata, which is itself one of the paper's
      findings. Operational values are drawn from a seeded distribution keyed to
      the server, so tools on the same server share a cost profile.

Queries are built from each target tool's own description with the tool name
removed, matching how the synthetic benchmark derives queries from the target.
Absolute recall is therefore optimistic for every policy in the same way; the
comparison between policies is what this experiment measures.
"""

import json
import math
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from hierarchical_discovery_benchmark import (
    CONSTRAINT_MARGIN, CONTEXT_OVERHEAD_TOKENS, ENERGY_WH_PER_TOKEN,
    EmbeddingPipeline, GHDHierarchy, count_tokens,
)

SEEDS = [42, 43, 44, 45, 46]
N_QUERIES = 300
N_CONSTRAINT_QUERIES = 200
GROUP_SIM_THRESHOLD = 0.82   # cosine on released embeddings, cross-server duplicates
DATA = Path("data")
OUT = Path("results")

FILLER = ["please", "for me", "right now", "if possible", "as soon as you can"]

# Query specificity. Long queries quote most of the target description and make
# the task close to lexical matching; short queries state intent only, which is
# closer to how an agent phrases a request.
QUERY_SPAN = (8, 18)

STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "with",
        "from", "by", "get", "this", "that", "it", "is", "are", "be"}


def load_catalogue():
    cat = json.loads((DATA / "mcp_tools_catalogue.json").read_text(encoding="utf-8"))
    emb = np.load(DATA / "mcp_tools_embeddings.npy")
    return cat, emb


def split_name(name):
    """Derive a verb and noun from a real tool name for cluster summaries."""
    parts = [p for p in re.split(r"[_\-\s./]+", name) if p]
    flat = []
    for p in parts:
        flat.extend(re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", p) or [p])
    flat = [w.lower() for w in flat if w]
    if not flat:
        return "invoke", name.lower()
    verb = flat[0]
    noun = " ".join(flat[1:]) or flat[0]
    return verb, noun


def build_entities(cat, seed):
    """Attach token counts and seeded operational profiles to the real tools."""
    rng = random.Random(seed)
    servers = sorted({t["server"] for t in cat})
    # One cost profile per server, so equivalent tools on different servers
    # genuinely differ on latency and energy, as they do in deployment.
    profile = {}
    for s in servers:
        r = random.Random(f"{seed}:{s}")
        profile[s] = (r.uniform(60, 420), r.uniform(0.15, 0.85))

    entities = []
    for i, t in enumerate(cat):
        verb, noun = split_name(t["name"])
        lat, en = profile[t["server"]]
        def_text = json.dumps({"name": t["name"], "description": t["description"],
                               "parameters": t["parameters"]}, ensure_ascii=False)
        entities.append({
            "idx": i,
            "name": t["name"],
            "description": t["description"] or t["name"],
            "server": t["server"],
            "verb": verb,
            "noun": noun,
            "kind": "mcp_tool",
            "latency_ms": lat * rng.uniform(0.85, 1.15),
            "energy_score": min(1.0, max(0.05, en * rng.uniform(0.85, 1.15))),
            "popularity": 1.0 / rng.randint(1, len(cat)),
            "def_text": def_text,
            "def_tokens": count_tokens(def_text),
        })
    return entities


def make_query(rng, e):
    """Query from the target's description, with the tool's own name removed."""
    name_words = set(re.split(r"[_\-\s./]+", e["name"].lower()))
    words = re.findall(r"[A-Za-z0-9']+", e["description"].lower())
    kept = [w for w in words if w not in name_words and w not in STOP]
    if len(kept) < 4:
        kept = [w for w in words if w not in STOP] or words
    span = kept[: rng.randint(*QUERY_SPAN)]
    return f"{rng.choice(FILLER)} {' '.join(span)}"


def build_groups(entities, emb):
    """Functionally equivalent tools that live on different servers.

    Equivalence is decided by cosine similarity on the released description
    embeddings, so the groups come from the real data rather than from a naming
    convention we impose.
    """
    E = normalize(emb.astype(np.float32))
    n = len(entities)
    servers = np.array([e["server"] for e in entities])
    groups, claimed = [], np.zeros(n, dtype=bool)
    # chunked similarity to keep memory bounded
    step = 512
    for start in range(0, n, step):
        block = E[start:start + step] @ E.T
        for r in range(block.shape[0]):
            i = start + r
            if claimed[i]:
                continue
            sims = block[r]
            cand = np.where(sims >= GROUP_SIM_THRESHOLD)[0]
            cand = [j for j in cand if not claimed[j]]
            if len(cand) < 3:
                continue
            if len(set(servers[cand])) < 3:
                continue
            for j in cand:
                claimed[j] = True
            groups.append(list(cand))
    return groups


def evaluate(seed, cat, emb, groups):
    rng = random.Random(seed + 1000)
    entities = build_entities(cat, seed)
    pipeline = EmbeddingPipeline()
    features, text_vecs = pipeline.fit_transform(entities)
    hierarchy = GHDHierarchy(entities, features, text_vecs)

    weights = [e["popularity"] for e in entities]
    targets = rng.choices(range(len(entities)), weights=weights, k=N_QUERIES)
    queries = [make_query(rng, entities[t]) for t in targets]
    q_vecs = pipeline.embed_queries(queries)

    flat_tokens = sum(e["def_tokens"] for e in entities) + CONTEXT_OVERHEAD_TOKENS
    res = {p: {"tokens": [], "hits": 0} for p in ["FLAT", "RET-5", "RET-B", "GHD", "GHD-NG"]}

    probe = [hierarchy.discover(q_vecs[i])[1] for i in range(min(60, N_QUERIES))]
    mean_def = statistics.mean(e["def_tokens"] for e in entities)
    k_budget = max(5, int((statistics.mean(probe) - CONTEXT_OVERHEAD_TOKENS) / mean_def))

    for i, t in enumerate(targets):
        qv = q_vecs[i]
        sims = text_vecs @ qv
        res["FLAT"]["tokens"].append(flat_tokens)
        res["FLAT"]["hits"] += 1
        top5 = np.argsort(-sims)[:5]
        res["RET-5"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in top5))
        res["RET-5"]["hits"] += int(t in top5)
        topb = np.argsort(-sims)[:k_budget]
        res["RET-B"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in topb))
        res["RET-B"]["hits"] += int(t in topb)
        ex, tk = hierarchy.discover(qv, use_graph=True)
        res["GHD"]["tokens"].append(tk)
        res["GHD"]["hits"] += int(t in ex)
        exng, tkng = hierarchy.discover(qv, use_graph=False)
        res["GHD-NG"]["tokens"].append(tkng)
        res["GHD-NG"]["hits"] += int(t in exng)

    # RET-B at GHD's marginal budget: if cluster summaries are treated as
    # persistent session context rather than a per-query cost, GHD's per-query
    # spend is only its injected definitions. This gives flat retrieval that same
    # per-query allowance so the two are compared on equal marginal terms.
    ghd_marginal = statistics.mean(res["GHD"]["tokens"]) - hierarchy.summary_tokens
    k_marg = max(5, int(ghd_marginal / mean_def))
    hits_marg = 0
    for i, t in enumerate(targets):
        sims = text_vecs @ q_vecs[i]
        hits_marg += int(t in np.argsort(-sims)[:k_marg])
    res_marg = {"k": k_marg, "recall": hits_marg / N_QUERIES}

    # --- operational-constraint routing on real functional groups -----------
    con = {"RET_top1": 0, "RETOP_top1": 0, "GHD_top1": 0,
           "RET_cover": 0, "RETOP_cover": 0, "GHD_cover": 0, "n": 0}
    if groups:
        for _ in range(N_CONSTRAINT_QUERIES):
            grp = rng.choice(groups)
            axis = rng.choice(["latency", "energy"])
            key = "latency_ms" if axis == "latency" else "energy_score"
            target = min(grp, key=lambda j: entities[j][key])
            base = entities[rng.choice(grp)]
            phrase = ("with the lowest latency" if axis == "latency"
                      else "that is the most energy efficient")
            q = f"{make_query(rng, base)}, choosing the option {phrase}"
            qv = pipeline.embed_queries([q])[0]
            sims = text_vecs @ qv
            op_all = np.array([e[key] for e in entities])
            con["n"] += 1

            topb = list(np.argsort(-sims)[:k_budget])
            con["RET_cover"] += int(target in topb)
            con["RET_top1"] += int(bool(topb) and topb[0] == target)

            cand = np.array(topb)
            cs = sims[cand]
            functional = cand[cs >= cs.max() - CONSTRAINT_MARGIN]
            functional = functional[np.argsort(op_all[functional])]
            rest = [j for j in cand if j not in set(functional.tolist())]
            order_op = list(functional) + rest
            con["RETOP_cover"] += int(target in order_op)
            con["RETOP_top1"] += int(bool(order_op) and order_op[0] == target)

            ex, _ = hierarchy.discover(qv, use_graph=True, op_pref=(axis, 1.0))
            con["GHD_cover"] += int(target in ex)
            con["GHD_top1"] += int(bool(ex) and ex[0] == target)

    out = {
        "seed": seed,
        "n_entities": len(entities),
        "n_clusters": len(hierarchy.cluster_ids),
        "summary_tokens": hierarchy.summary_tokens,
        "flat_tokens": flat_tokens,
        "k_budget_matched": k_budget,
        "policies": {},
        "constraint": {},
        "retb_at_ghd_marginal_budget": res_marg,
    }
    out["summary_tokens"] = hierarchy.summary_tokens
    out["mean_def_tokens"] = mean_def
    for p, r in res.items():
        mt = statistics.mean(r["tokens"])
        out["policies"][p] = {
            "mean_tokens": mt,
            "marginal_tokens": mt - hierarchy.summary_tokens if p.startswith("GHD") else mt,
            "recall": r["hits"] / N_QUERIES,
            "token_reduction_vs_flat_percent": (1 - mt / flat_tokens) * 100,
            "energy_wh_per_million_queries": mt * ENERGY_WH_PER_TOKEN * 1_000_000,
        }
    if con["n"]:
        for tag in ("RET", "RETOP", "GHD"):
            out["constraint"][f"{tag.lower()}_top1_recall"] = con[f"{tag}_top1"] / con["n"]
            out["constraint"][f"{tag.lower()}_coverage"] = con[f"{tag}_cover"] / con["n"]
        out["constraint"]["n"] = con["n"]
    return out


def main():
    cat, emb = load_catalogue()
    print("=" * 70)
    print("GHD on the real MCP-tools catalogue (MCP-Zero, arXiv:2506.01056)")
    print("=" * 70)
    servers = len({t["server"] for t in cat})
    print(f"catalogue: {len(cat)} tools across {servers} servers")

    t0 = time.perf_counter()
    groups = build_groups(build_entities(cat, 42), emb)
    print(f"functional-equivalence groups (cross-server, cosine >= {GROUP_SIM_THRESHOLD}): "
          f"{len(groups)} groups covering {sum(len(g) for g in groups)} tools "
          f"({time.perf_counter() - t0:.1f}s)")

    reps = []
    for s in SEEDS:
        r = evaluate(s, cat, emb, groups)
        reps.append(r)
        print(f"  seed {s}: {r['n_clusters']} clusters, GHD tokens "
              f"{r['policies']['GHD']['mean_tokens']:.0f}, "
              f"GHD recall {r['policies']['GHD']['recall'] * 100:.1f}%")

    agg = {"dataset": "MCP-tools (MCP-Zero, arXiv:2506.01056)",
           "n_tools": len(cat), "n_servers": servers,
           "seeds": SEEDS, "repetitions": reps, "policies": {}, "constraint": {}}
    for p in reps[0]["policies"]:
        agg["policies"][p] = {
            "mean_tokens": statistics.mean(r["policies"][p]["mean_tokens"] for r in reps),
            "std_tokens": statistics.stdev(r["policies"][p]["mean_tokens"] for r in reps),
            "recall_mean": statistics.mean(r["policies"][p]["recall"] for r in reps),
            "recall_std": statistics.stdev(r["policies"][p]["recall"] for r in reps),
            "token_reduction_vs_flat_percent": statistics.mean(
                r["policies"][p]["token_reduction_vs_flat_percent"] for r in reps),
            "energy_wh_per_million_queries": statistics.mean(
                r["policies"][p]["energy_wh_per_million_queries"] for r in reps),
        }
    if reps[0]["constraint"]:
        for k in reps[0]["constraint"]:
            if k == "n":
                continue
            vals = [r["constraint"][k] for r in reps]
            agg["constraint"][k + "_mean"] = statistics.mean(vals)
            agg["constraint"][k + "_std"] = statistics.stdev(vals)
    agg["retb_at_ghd_marginal_budget"] = {
        "k_mean": statistics.mean(r["retb_at_ghd_marginal_budget"]["k"] for r in reps),
        "recall_mean": statistics.mean(r["retb_at_ghd_marginal_budget"]["recall"] for r in reps),
        "recall_std": statistics.stdev(r["retb_at_ghd_marginal_budget"]["recall"] for r in reps),
    }
    agg["summary_tokens_mean"] = statistics.mean(r["summary_tokens"] for r in reps)
    agg["n_groups"] = len(groups)
    agg["n_grouped_tools"] = sum(len(g) for g in groups)
    agg["flat_tokens_mean"] = statistics.mean(r["flat_tokens"] for r in reps)
    agg["n_clusters_mean"] = statistics.mean(r["n_clusters"] for r in reps)

    OUT.mkdir(exist_ok=True)
    path = OUT / "real_catalogue_results.json"
    path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")

    print("\nSUMMARY (mean over 5 seeds, 2,797 real tools)")
    for p, v in agg["policies"].items():
        print(f"  {p:7s} tokens={v['mean_tokens']:>9,.0f}  recall={v['recall_mean'] * 100:5.1f}%"
              f"  reduction vs flat={v['token_reduction_vs_flat_percent']:5.1f}%")
    m = agg["retb_at_ghd_marginal_budget"]
    print(f"  GHD summaries are persistent context ({agg['summary_tokens_mean']:.0f} tokens).")
    print(f"  At equal MARGINAL budget (k={m['k_mean']:.0f} definitions each): "
          f"RET-B {m['recall_mean'] * 100:.1f}%  vs  GHD "
          f"{agg['policies']['GHD']['recall_mean'] * 100:.1f}%")
    if agg["constraint"]:
        c = agg["constraint"]
        print(f"  constraint routing top-1: RET-B {c['ret_top1_recall_mean'] * 100:.1f}%"
              f"  RET-OP {c['retop_top1_recall_mean'] * 100:.1f}%"
              f"  GHD {c['ghd_top1_recall_mean'] * 100:.1f}%")


if __name__ == "__main__":
    main()
