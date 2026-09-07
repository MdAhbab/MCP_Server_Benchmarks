"""
Ablation of the EATS selection layer on the real MCP catalogue.

Runs the stage ladder R, RA, RAP, RAPC, RAPCB, RAPCBL under both encoders used
elsewhere in this suite: the deterministic term-frequency pipeline and the
neural sentence encoder. Reporting both matters because the stage the layer
replaces, a fixed similarity margin, is exactly the component that failed to
transfer between them.

Metrics per rung:
  recall          the correct tool is exposed to the model
  routing top-1   among functionally equivalent tools, the operationally
                  cheapest one is ranked first
  tokens/query    definition tokens injected, the quantity that becomes
                  prefill energy

The learned stage is trained on held-out seeds. No query it scores was used to
fit it.

    python eats_benchmark.py              # both encoders
    python eats_benchmark.py --tfidf-only # skip the neural encoder
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np

import real_catalogue_benchmark as R
from eats_selection import EATSSelector, LearnedReranker, build_training_set
from hierarchical_discovery_benchmark import (
    CONSTRAINT_MARGIN, CONTEXT_OVERHEAD_TOKENS, EmbeddingPipeline,
)

SEEDS = [42, 43, 44, 45, 46]
# The learned stage is evaluated leave-one-seed-out: for each test seed it is
# fitted on the other four, so no query it scores was used to fit it.
N_QUERIES = 300
N_CONSTRAINT = 200
# Three band rules are compared head to head, because the band is the component
# that failed to transfer between encoders and is the one stage A replaces.
LADDER = [
    ("R",           "R",      "relative"),   # retrieval only
    ("C fixed",     "RC",     "fixed"),      # cost-aware over a hand-tuned absolute margin
    ("C gap",       "RAC",    "adaptive"),   # band from the largest gap in the curve
    ("C relative",  "RAC",    "relative"),   # band as a fraction of the query's spread
    ("+Pareto",     "RAPC",   "relative"),
    ("+Budget",     "RAPCB",  "relative"),
    ("+Learned",    "RAPCBL", "relative"),
]
OUT = Path("results")


def constraint_queries(rng, entities, groups, n):
    """Queries that name an operational preference among equivalent tools."""
    qs, targets, axes = [], [], []
    for _ in range(n):
        grp = rng.choice(groups)
        axis = rng.choice(["latency", "energy"])
        key = "latency_ms" if axis == "latency" else "energy_score"
        target = min(grp, key=lambda j: entities[j][key])
        base = entities[rng.choice(grp)]
        phrase = ("with the lowest latency" if axis == "latency"
                  else "that is the most energy efficient")
        qs.append(f"{R.make_query(rng, base)}, choosing the option {phrase}")
        targets.append(target)
        axes.append(axis)
    return qs, targets, axes


def fixed_margin_top1(sims_all, cand, entities, target, axis):
    """The paper's earlier rule: fixed similarity band, then cheapest first.

    Kept as the comparison point for stage A, since this is the component the
    adaptive band replaces.
    """
    key = "latency_ms" if axis == "latency" else "energy_score"
    cost = np.array([entities[i][key] for i in cand])
    s = sims_all[cand]
    band = cost[s >= s.max() - CONSTRAINT_MARGIN]
    idx = np.where(s >= s.max() - CONSTRAINT_MARGIN)[0]
    if idx.size == 0:
        return False
    ordered = idx[np.argsort(cost[idx])]
    return int(cand[ordered[0]]) == target


def run_encoder(label, cat, emb, groups, encode_fn, tfidf=True):
    per_seed = []
    for seed in SEEDS:
        rng = random.Random(seed + 1000)
        entities = R.build_entities(cat, seed)

        if tfidf:
            pipe = EmbeddingPipeline()
            _, text_vecs = pipe.fit_transform(entities)
            embed = pipe.embed_queries
        else:
            text_vecs = encode_fn["tools"]
            embed = encode_fn["queries"]

        weights = [e["popularity"] for e in entities]
        targets = rng.choices(range(len(entities)), weights=weights, k=N_QUERIES)
        q_vecs = np.asarray(embed([R.make_query(rng, entities[t]) for t in targets]))

        cq, ct, ca = constraint_queries(rng, entities, groups, N_CONSTRAINT)
        cq_vecs = np.asarray(embed(cq))

        mean_def = statistics.mean(e["def_tokens"] for e in entities)
        k_max = max(5, int(4000 / mean_def))

        rr = None
        if True:
            # leave-one-seed-out: fit on every seed except this one
            Xs, ys = [], []
            for ts in [t for t in SEEDS if t != seed]:
                trng = random.Random(ts + 1000)
                tent = R.build_entities(cat, ts)
                if tfidf:
                    tp = EmbeddingPipeline()
                    _, tvec = tp.fit_transform(tent)
                    temb = tp.embed_queries
                else:
                    tvec, temb = encode_fn["tools"], encode_fn["queries"]
                tw = [e["popularity"] for e in tent]
                _ = trng.choices(range(len(tent)), weights=tw, k=N_QUERIES)
                tcq, tct, tca = constraint_queries(trng, tent, groups, N_CONSTRAINT)
                tsel = EATSSelector(tent, tvec, k_max)
                X, y = build_training_set(tsel, np.asarray(temb(tcq)), tct, tca)
                if X.size:
                    Xs.append(X)
                    ys.append(y)
            if Xs:
                rr = LearnedReranker().fit(np.vstack(Xs), np.concatenate(ys))

        sel = EATSSelector(entities, text_vecs, k_max, reranker=rr)

        row = {"seed": seed, "k_max": k_max, "stages": {}}
        # fixed-margin reference for stage A
        fm = 0
        for i, t in enumerate(ct):
            sims_all = text_vecs @ cq_vecs[i]
            cand = np.argsort(-sims_all)[:k_max]
            fm += int(fixed_margin_top1(sims_all, cand, entities, t, ca[i]))
        row["fixed_margin_top1"] = fm / len(ct)

        for label, stages, band in LADDER:
            if "L" in stages and rr is None:
                continue
            hits = toks = 0
            for i, t in enumerate(targets):
                ex, tk = sel.select(q_vecs[i], axis=None, stages=stages, band=band)
                hits += int(t in ex)
                toks += tk + CONTEXT_OVERHEAD_TOKENS
            c1 = ctoks = 0
            for i, t in enumerate(ct):
                ex, tk = sel.select(cq_vecs[i], axis=ca[i], stages=stages, band=band)
                c1 += int(bool(ex) and ex[0] == t)
                ctoks += tk + CONTEXT_OVERHEAD_TOKENS
            row["stages"][label] = {
                "recall": hits / N_QUERIES,
                "tokens": toks / N_QUERIES,
                "tokens_constraint": ctoks / len(ct),
                "routing_top1": c1 / len(ct),
            }
        per_seed.append(row)
        done = ", ".join(f"{k}:{v['routing_top1']*100:.0f}%"
                         for k, v in row["stages"].items())
        print(f"  seed {seed}: {done}")

    agg = {}
    for label, _st, _bd in LADDER:
        vals = [r["stages"][label] for r in per_seed if label in r["stages"]]
        if not vals:
            continue
        agg[label] = {
            "recall_mean": statistics.mean(v["recall"] for v in vals),
            "recall_std": statistics.stdev(v["recall"] for v in vals) if len(vals) > 1 else 0.0,
            "tokens_mean": statistics.mean(v["tokens"] for v in vals),
            "tokens_constraint_mean": statistics.mean(v["tokens_constraint"] for v in vals),
            "routing_top1_mean": statistics.mean(v["routing_top1"] for v in vals),
            "routing_top1_std": statistics.stdev(v["routing_top1"] for v in vals) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        }
    fmv = [r["fixed_margin_top1"] for r in per_seed]
    return {"encoder": label, "per_seed": per_seed, "aggregate": agg,
            "fixed_margin_top1_mean": statistics.mean(fmv),
            "fixed_margin_top1_std": statistics.stdev(fmv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfidf-only", action="store_true")
    args = ap.parse_args()

    cat, emb = R.load_catalogue()
    groups = R.build_groups(R.build_entities(cat, 42), emb)
    print("=" * 72)
    print("EATS ablation on the real MCP catalogue")
    print("=" * 72)
    print(f"catalogue: {len(cat)} tools | equivalence groups: {len(groups)} "
          f"covering {sum(len(g) for g in groups)} tools")

    out = {"n_tools": len(cat), "n_groups": len(groups),
           "ladder": [l for l, _s, _b in LADDER],
           "learned_stage_protocol": "leave-one-seed-out", "encoders": {}}

    print("\nterm-frequency encoder")
    out["encoders"]["tfidf"] = run_encoder("tfidf", cat, emb, groups, None, tfidf=True)

    if not args.tfidf_only:
        print("\nneural encoder")
        import dense_catalogue_benchmark as D
        servers = json.loads((Path("data") / "mcp_servers.json").read_text(encoding="utf-8"))
        tools_e, _, _ = D.encode_corpus(cat, servers)
        fn = {"tools": tools_e, "queries": D.encode_queries}
        out["encoders"]["neural"] = run_encoder("neural", cat, emb, groups, fn, tfidf=False)

    OUT.mkdir(exist_ok=True)
    (OUT / "eats_ablation_results.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    print(f"\nResults saved to {OUT / 'eats_ablation_results.json'}")

    for enc, res in out["encoders"].items():
        print(f"\n{enc.upper()} (mean over {len(SEEDS)} seeds)")
        print(f"  fixed margin (prior rule)  routing {res['fixed_margin_top1_mean']*100:5.1f}%")
        for stages, v in res["aggregate"].items():
            print(f"  {stages:13s} recall={v['recall_mean']*100:5.1f}%  "
                  f"tok={v['tokens_mean']:6.0f}  tok_con={v['tokens_constraint_mean']:6.0f}  "
                  f"routing={v['routing_top1_mean']*100:5.1f}% "
                  f"(SD {v['routing_top1_std']*100:4.1f})")


if __name__ == "__main__":
    main()
