"""
Tool-discovery policies on the real MCP catalogue with a dense encoder.

This extends real_catalogue_benchmark.py in two ways the deterministic
TF-IDF pipeline could not cover.

1. A neural sentence encoder replaces TF-IDF, so absolute recall is not
   limited by lexical matching. Every policy shares one embedding space, so
   the comparison between policies stays internally fair.

2. Two published systems are reimplemented from their own descriptions rather
   than approximated by generic retrieval:

   RAG-MCP (Gan and Sun, arXiv:2505.03275) indexes tool definitions externally
   and injects the top-k semantic matches.

   MCP-Zero (Fei, Zheng and Feng, arXiv:2506.01056) routes in two stages. Their
   released matcher scores each server by max(cos(q, server description),
   cos(q, server summary)), keeps the top 5 servers, then returns the top 3
   tools within them. We follow that procedure exactly, and additionally report
   a budget-matched variant so the token axis is comparable.

Fidelity note: MCP-Zero's published numbers use OpenAI text-embedding-3-large.
That model cannot be queried offline, so we reimplement the algorithm and run it
in the local encoder's space alongside every other policy. What is compared here
is the routing procedure, not the embedding model.

Prerequisites:
    pip install sentence-transformers
    python mcp_zero_catalogue.py <downloaded.json>

Run:
    python dense_catalogue_benchmark.py
"""

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from hierarchical_discovery_benchmark import (
    CONSTRAINT_MARGIN, CONTEXT_OVERHEAD_TOKENS, EXPAND_MEMBER_BUDGET, SVD_DIM,
    GHDHierarchy, count_tokens,
)
import real_catalogue_benchmark as R

MODEL = "BAAI/bge-small-en-v1.5"
SEEDS = [42, 43, 44, 45, 46]
N_QUERIES = 300
N_CONSTRAINT_QUERIES = 200
TOP_SERVERS = 5      # MCP-Zero default
TOP_TOOLS = 3        # MCP-Zero default
N_KMEANS_CLUSTERS = 90   # matches the cluster count TF-IDF/HDBSCAN finds here
DATA = Path("data")
OUT = Path("results")
CACHE = DATA / "dense_embeddings.npz"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_corpus(cat, servers):
    """Embed tools, server descriptions, and server summaries once, then cache."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["tools"], z["srv_desc"], z["srv_summ"]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    tool_texts = [f"{t['name'].replace('_', ' ')}. {t['description']}" for t in cat]
    t0 = time.perf_counter()
    tools = m.encode(tool_texts, normalize_embeddings=True, batch_size=128,
                     show_progress_bar=False).astype(np.float32)
    srv_desc = m.encode([s["description"] or s["name"] for s in servers],
                        normalize_embeddings=True, batch_size=128,
                        show_progress_bar=False).astype(np.float32)
    srv_summ = m.encode([s["summary"] or s["description"] or s["name"] for s in servers],
                        normalize_embeddings=True, batch_size=128,
                        show_progress_bar=False).astype(np.float32)
    print(f"  encoded {len(cat)} tools and {len(servers)} servers "
          f"in {time.perf_counter() - t0:.1f}s (dim {tools.shape[1]})")
    np.savez_compressed(CACHE, tools=tools, srv_desc=srv_desc, srv_summ=srv_summ)
    return tools, srv_desc, srv_summ


def encode_queries(queries):
    from sentence_transformers import SentenceTransformer
    global _QM
    try:
        m = _QM
    except NameError:
        m = _QM = SentenceTransformer(MODEL)
    return m.encode(queries, normalize_embeddings=True, batch_size=128,
                    show_progress_bar=False).astype(np.float32)


class DensePipeline:
    """Adapter so GHDHierarchy can consume precomputed dense text vectors.

    GHD's own pipeline concatenates standardised operational features onto the
    semantic block. That behaviour is preserved here; only the semantic block
    changes from TF-IDF/SVD to the neural encoder.

    The clustering features are reduced to SVD_DIM components first. Density
    clustering degrades badly at the encoder's native 384 dimensions: run
    directly on the full vectors, HDBSCAN returns two clusters for the whole
    2,797-tool catalogue, which makes the hierarchy degenerate into flat
    retrieval. Reducing to the same dimensionality the TF-IDF pipeline uses
    keeps the two configurations comparable. Retrieval and centroid similarity
    still use the full-dimensional vectors.
    """

    def __init__(self, text_vecs, op_weight=0.05, svd_dim=SVD_DIM):
        self.text_vecs = text_vecs
        self.op_weight = op_weight
        self.svd_dim = svd_dim

    def features(self, entities):
        import math
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler, normalize as _norm
        reduced = _norm(TruncatedSVD(n_components=self.svd_dim,
                                     random_state=0).fit_transform(self.text_vecs))
        ops = np.array([[math.log(e["latency_ms"]), e["energy_score"],
                         math.log(e["popularity"]),
                         1.0 if e["kind"] == "a2a_capability" else 0.0]
                        for e in entities])
        ops = StandardScaler().fit_transform(ops) * self.op_weight
        return np.hstack([reduced, ops])


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def mcp_zero_route(qv, tools_e, srv_desc, srv_summ, server_of, top_servers=TOP_SERVERS,
                   top_tools=TOP_TOOLS):
    """MCP-Zero's two-stage router, following their released matcher."""
    srv_score = np.maximum(srv_desc @ qv, srv_summ @ qv)
    chosen = set(np.argsort(-srv_score)[:top_servers].tolist())
    pool = np.where(np.isin(server_of, list(chosen)))[0]
    if pool.size == 0:
        return []
    sims = tools_e[pool] @ qv
    return [int(pool[j]) for j in np.argsort(-sims)[:top_tools]]


def evaluate(seed, cat, servers, tools_e, srv_desc, srv_summ, groups):
    rng = random.Random(seed + 1000)
    entities = R.build_entities(cat, seed)
    srv_index = {s["name"]: i for i, s in enumerate(servers)}
    server_of = np.array([srv_index.get(e["server"], 0) for e in entities])

    pipe = DensePipeline(tools_e)
    feats = pipe.features(entities)
    # GHD as proposed (HDBSCAN), and GHD with a partitioning clusterer matched to
    # the cluster count the TF-IDF configuration discovers on this catalogue.
    hierarchy = GHDHierarchy(entities, feats, tools_e)
    from sklearn.cluster import KMeans
    hierarchy_km = GHDHierarchy(entities, feats, tools_e,
                                clusterer=KMeans(n_clusters=N_KMEANS_CLUSTERS,
                                                 random_state=seed, n_init=4))

    weights = [e["popularity"] for e in entities]
    targets = rng.choices(range(len(entities)), weights=weights, k=N_QUERIES)
    q_vecs = encode_queries([R.make_query(rng, entities[t]) for t in targets])

    flat_tokens = sum(e["def_tokens"] for e in entities) + CONTEXT_OVERHEAD_TOKENS
    mean_def = statistics.mean(e["def_tokens"] for e in entities)

    probe = [hierarchy.discover(q_vecs[i])[1] for i in range(min(60, N_QUERIES))]
    k_budget = max(5, int((statistics.mean(probe) - CONTEXT_OVERHEAD_TOKENS) / mean_def))

    SERVER_SWEEP = [5, 10, 25, 50]
    names = (["FLAT", "RAG-MCP", "MCP-Zero", "MCP-Zero-B", "GHD", "GHD-NG"] +
             [f"MCP-Zero-s{k}" for k in SERVER_SWEEP if k != TOP_SERVERS] +
             ["GHD-KMeans"])
    res = {p: {"tokens": [], "hits": 0} for p in names}

    for i, t in enumerate(targets):
        qv = q_vecs[i]
        sims = tools_e @ qv

        res["FLAT"]["tokens"].append(flat_tokens)
        res["FLAT"]["hits"] += 1

        topb = np.argsort(-sims)[:k_budget]
        res["RAG-MCP"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in topb))
        res["RAG-MCP"]["hits"] += int(t in topb)

        mz = mcp_zero_route(qv, tools_e, srv_desc, srv_summ, server_of)
        res["MCP-Zero"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in mz))
        res["MCP-Zero"]["hits"] += int(t in mz)

        mzb = mcp_zero_route(qv, tools_e, srv_desc, srv_summ, server_of,
                             top_tools=EXPAND_MEMBER_BUDGET)
        res["MCP-Zero-B"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in mzb))
        res["MCP-Zero-B"]["hits"] += int(t in mzb)

        for k in SERVER_SWEEP:
            if k == TOP_SERVERS:
                continue
            mzk = mcp_zero_route(qv, tools_e, srv_desc, srv_summ, server_of,
                                 top_servers=k, top_tools=EXPAND_MEMBER_BUDGET)
            res[f"MCP-Zero-s{k}"]["tokens"].append(
                CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in mzk))
            res[f"MCP-Zero-s{k}"]["hits"] += int(t in mzk)

        ex, tk = hierarchy.discover(qv, use_graph=True)
        res["GHD"]["tokens"].append(tk)
        res["GHD"]["hits"] += int(t in ex)

        exn, tkn = hierarchy.discover(qv, use_graph=False)
        res["GHD-NG"]["tokens"].append(tkn)
        res["GHD-NG"]["hits"] += int(t in exn)

        exk, tkk = hierarchy_km.discover(qv, use_graph=True)
        res["GHD-KMeans"]["tokens"].append(tkk)
        res["GHD-KMeans"]["hits"] += int(t in exk)

    # RAG-MCP restricted to GHD's marginal definition budget
    ghd_marginal = statistics.mean(res["GHD"]["tokens"]) - hierarchy.summary_tokens
    k_marg = max(5, int(ghd_marginal / mean_def))
    hits_marg = sum(int(t in np.argsort(-(tools_e @ q_vecs[i]))[:k_marg])
                    for i, t in enumerate(targets))

    # --- operational-constraint routing --------------------------------
    con = {k: 0 for k in ("RAG_top1", "RAGOP_top1", "MZ_top1", "GHD_top1", "GHDKM_top1")}
    con["n"] = 0
    for _ in range(N_CONSTRAINT_QUERIES):
        grp = rng.choice(groups)
        axis = rng.choice(["latency", "energy"])
        key = "latency_ms" if axis == "latency" else "energy_score"
        target = min(grp, key=lambda j: entities[j][key])
        base = entities[rng.choice(grp)]
        phrase = ("with the lowest latency" if axis == "latency"
                  else "that is the most energy efficient")
        qv = encode_queries([f"{R.make_query(rng, base)}, choosing the option {phrase}"])[0]
        sims = tools_e @ qv
        op_all = np.array([e[key] for e in entities])
        con["n"] += 1

        topb = list(np.argsort(-sims)[:k_budget])
        con["RAG_top1"] += int(bool(topb) and topb[0] == target)

        cand = np.array(topb)
        cs = sims[cand]
        functional = cand[cs >= cs.max() - CONSTRAINT_MARGIN]
        functional = functional[np.argsort(op_all[functional])]
        rest = [j for j in cand if j not in set(functional.tolist())]
        order_op = list(functional) + rest
        con["RAGOP_top1"] += int(bool(order_op) and order_op[0] == target)

        mz = mcp_zero_route(qv, tools_e, srv_desc, srv_summ, server_of)
        con["MZ_top1"] += int(bool(mz) and mz[0] == target)

        ex, _ = hierarchy.discover(qv, use_graph=True, op_pref=(axis, 1.0))
        con["GHD_top1"] += int(bool(ex) and ex[0] == target)

        exk, _ = hierarchy_km.discover(qv, use_graph=True, op_pref=(axis, 1.0))
        con["GHDKM_top1"] += int(bool(exk) and exk[0] == target)

    out = {"seed": seed, "n_clusters": len(hierarchy.cluster_ids),
           "n_clusters_kmeans": len(hierarchy_km.cluster_ids),
           "summary_tokens": hierarchy.summary_tokens, "k_budget": k_budget,
           "policies": {}, "constraint": {},
           "ragmcp_at_ghd_marginal": {"k": k_marg, "recall": hits_marg / N_QUERIES}}
    for p, r in res.items():
        mt = statistics.mean(r["tokens"])
        out["policies"][p] = {"mean_tokens": mt, "recall": r["hits"] / N_QUERIES,
                              "token_reduction_vs_flat_percent": (1 - mt / flat_tokens) * 100}
    for tag, key in (("RAG_top1", "ragmcp_top1"), ("RAGOP_top1", "ragmcp_op_top1"),
                     ("MZ_top1", "mcpzero_top1"), ("GHD_top1", "ghd_top1"),
                     ("GHDKM_top1", "ghd_kmeans_top1")):
        out["constraint"][key] = con[tag] / con["n"]
    return out


def main():
    cat, emb = R.load_catalogue()
    servers = json.loads((DATA / "mcp_servers.json").read_text(encoding="utf-8"))
    print("=" * 72)
    print("Discovery policies on the real MCP catalogue, dense encoder")
    print("=" * 72)
    print(f"catalogue: {len(cat)} tools, {len(servers)} servers | encoder: {MODEL}")

    tools_e, srv_desc, srv_summ = encode_corpus(cat, servers)
    groups = R.build_groups(R.build_entities(cat, 42), emb)
    print(f"functional-equivalence groups: {len(groups)} covering "
          f"{sum(len(g) for g in groups)} tools")

    reps = []
    for s in SEEDS:
        r = evaluate(s, cat, servers, tools_e, srv_desc, srv_summ, groups)
        reps.append(r)
        print(f"  seed {s}: {r['n_clusters']} clusters, "
              f"GHD recall {r['policies']['GHD']['recall'] * 100:.1f}%")

    agg = {"encoder": MODEL, "n_tools": len(cat), "n_servers": len(servers),
           "seeds": SEEDS, "repetitions": reps, "policies": {}, "constraint": {}}
    for p in reps[0]["policies"]:
        agg["policies"][p] = {
            "mean_tokens": statistics.mean(r["policies"][p]["mean_tokens"] for r in reps),
            "recall_mean": statistics.mean(r["policies"][p]["recall"] for r in reps),
            "recall_std": statistics.stdev(r["policies"][p]["recall"] for r in reps),
            "token_reduction_vs_flat_percent": statistics.mean(
                r["policies"][p]["token_reduction_vs_flat_percent"] for r in reps),
        }
    for k in reps[0]["constraint"]:
        vals = [r["constraint"][k] for r in reps]
        agg["constraint"][k + "_mean"] = statistics.mean(vals)
        agg["constraint"][k + "_std"] = statistics.stdev(vals)
    agg["ragmcp_at_ghd_marginal"] = {
        "k_mean": statistics.mean(r["ragmcp_at_ghd_marginal"]["k"] for r in reps),
        "recall_mean": statistics.mean(r["ragmcp_at_ghd_marginal"]["recall"] for r in reps),
    }
    agg["summary_tokens_mean"] = statistics.mean(r["summary_tokens"] for r in reps)

    OUT.mkdir(exist_ok=True)
    (OUT / "dense_catalogue_results.json").write_text(json.dumps(agg, indent=2),
                                                      encoding="utf-8")
    print(f"\nResults saved to {OUT / 'dense_catalogue_results.json'}")

    print("\nSUMMARY (mean over 5 seeds, dense encoder)")
    for p, v in agg["policies"].items():
        print(f"  {p:11s} tokens={v['mean_tokens']:>9,.0f}  "
              f"recall={v['recall_mean'] * 100:5.1f}% "
              f"(SD {v['recall_std'] * 100:4.1f})")
    m = agg["ragmcp_at_ghd_marginal"]
    print(f"  RAG-MCP at GHD's marginal budget (k={m['k_mean']:.0f}): "
          f"{m['recall_mean'] * 100:.1f}%")
    c = agg["constraint"]
    print(f"\n  constraint routing top-1:")
    print(f"    RAG-MCP           {c['ragmcp_top1_mean'] * 100:5.1f}%")
    print(f"    RAG-MCP + op rank {c['ragmcp_op_top1_mean'] * 100:5.1f}%")
    print(f"    MCP-Zero          {c['mcpzero_top1_mean'] * 100:5.1f}%")
    print(f"    GHD (HDBSCAN)     {c['ghd_top1_mean'] * 100:5.1f}%")
    print(f"    GHD (KMeans)      {c['ghd_kmeans_top1_mean'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
