"""
Scores the artefacts run.py generated.

run.py uses the GPU to produce two things that could not be produced without a
model: natural-language queries written in a user's voice, and server
descriptions of the kind MCP-Zero's first stage expects. This script evaluates
the discovery policies on them.

  E3  discovery under natural-language queries, against the description-derived
      queries used elsewhere, so the optimism of that construction is quantified
  E4  MCP-Zero's router with a model-written server description for stage one
  E5  the EATS ablation on the natural queries, converted to joules per query
      using the measured prefill cost per token from E2

    python gpu_followups.py
"""

import json
import statistics
from pathlib import Path

import numpy as np

import real_catalogue_benchmark as R
from eats_selection import EATSSelector, LearnedReranker, build_training_set
from hierarchical_discovery_benchmark import CONTEXT_OVERHEAD_TOKENS

GPU = Path("results/gpu")
DATA = Path("data")
SEED = 42


def need(p: Path):
    if not p.exists():
        raise SystemExit(f"missing {p}. Run run.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


def encode(texts):
    import dense_catalogue_benchmark as D
    return np.asarray(D.encode_queries(list(texts)))


def main():
    cat, emb = R.load_catalogue()
    servers = json.loads((DATA / "mcp_servers.json").read_text(encoding="utf-8"))
    nat = need(GPU / "e3_natural_queries.json")
    entities = R.build_entities(cat, SEED)

    import dense_catalogue_benchmark as D
    tools_e, srv_desc_e, srv_summ_e = D.encode_corpus(cat, servers)

    targets = nat["targets"]
    q_nat = encode(nat["queries"])

    # description-derived queries for the same targets, as the comparison point
    import random
    rng = random.Random(SEED + 1000)
    q_desc = encode([R.make_query(rng, entities[t]) for t in targets])

    mean_def = statistics.mean(e["def_tokens"] for e in entities)
    k = max(5, int(4000 / mean_def))
    srv_index = {s["name"]: i for i, s in enumerate(servers)}
    server_of = np.array([srv_index.get(e["server"], 0) for e in entities])

    out = {"n_queries": len(targets), "k_budget": k}

    # ---- E3: query construction ------------------------------------------
    def recall_at(qv_all, policy):
        hits = 0
        for i, t in enumerate(targets):
            qv = qv_all[i]
            if policy == "ragmcp":
                hits += int(t in np.argsort(-(tools_e @ qv))[:k])
            else:
                ex = D.mcp_zero_route(qv, tools_e, srv_desc_e, srv_summ_e, server_of)
                hits += int(t in ex)
        return hits / len(targets)

    out["e3_query_construction"] = {
        "ragmcp_description_derived": recall_at(q_desc, "ragmcp"),
        "ragmcp_natural": recall_at(q_nat, "ragmcp"),
        "mcpzero_description_derived": recall_at(q_desc, "mcpzero"),
        "mcpzero_natural": recall_at(q_nat, "mcpzero"),
    }
    e3 = out["e3_query_construction"]
    print(f"E3 recall  RAG-MCP: description-derived {e3['ragmcp_description_derived']*100:.1f}% "
          f"-> natural {e3['ragmcp_natural']*100:.1f}%")

    # ---- E4: MCP-Zero with a model-written server description -------------
    sd = GPU / "e4_server_descriptions.json"
    if sd.exists():
        d = json.loads(sd.read_text(encoding="utf-8"))
        names, descs = d["servers"], d["descriptions"]
        gen_e = encode(descs)
        full = srv_desc_e.copy()
        for nm, vec in zip(names, gen_e):
            if nm in srv_index:
                full[srv_index[nm]] = vec
        hits_gen = hits_pub = 0
        for i, t in enumerate(targets):
            qv = q_nat[i]
            hits_pub += int(t in D.mcp_zero_route(qv, tools_e, srv_desc_e, srv_summ_e, server_of))
            hits_gen += int(t in D.mcp_zero_route(qv, full, srv_summ_e, srv_summ_e, server_of))
        out["e4_mcpzero_server_stage"] = {
            "n_servers_regenerated": len(names),
            "recall_published_descriptions": hits_pub / len(targets),
            "recall_model_written_descriptions": hits_gen / len(targets),
        }
        e4 = out["e4_mcpzero_server_stage"]
        print(f"E4 MCP-Zero  published {e4['recall_published_descriptions']*100:.1f}% "
              f"-> model-written {e4['recall_model_written_descriptions']*100:.1f}%")

    # ---- E5: EATS on natural queries, priced in joules --------------------
    groups = R.build_groups(entities, emb)
    cq, ct, ca = [], [], []
    for _ in range(200):
        grp = rng.choice(groups)
        axis = rng.choice(["latency", "energy"])
        key = "latency_ms" if axis == "latency" else "energy_score"
        ct.append(min(grp, key=lambda j: entities[j][key]))
        base = entities[rng.choice(grp)]
        phrase = ("with the lowest latency" if axis == "latency"
                  else "that is the most energy efficient")
        cq.append(f"{R.make_query(rng, base)}, choosing the option {phrase}")
        ca.append(axis)
    cq_v = encode(cq)

    sel_train = EATSSelector(entities, tools_e, k)
    X, y = build_training_set(sel_train, cq_v[:100], ct[:100], ca[:100])
    rr = LearnedReranker().fit(X, y) if X.size else None
    sel = EATSSelector(entities, tools_e, k, reranker=rr)

    # joules per input token, measured in E2
    jpt = None
    p2 = GPU / "e2_prefill_decode.json"
    if p2.exists():
        rows = [r for r in json.loads(p2.read_text(encoding="utf-8"))["rows"]
                if r.get("prefill_j_per_token")]
        if rows:
            jpt = statistics.mean(r["prefill_j_per_token"] for r in rows)

    ladder = [("R", "R", "relative"), ("C relative", "RAC", "relative"),
              ("+Pareto", "RAPC", "relative"), ("+Budget", "RAPCB", "relative"),
              ("+Learned", "RAPCBL", "relative")]
    rung = {}
    for label, stages, band in ladder:
        if "L" in stages and rr is None:
            continue
        hits = toks = 0
        for i, t in enumerate(targets):
            ex, tk = sel.select(q_nat[i], axis=None, stages=stages, band=band)
            hits += int(t in ex)
            toks += tk + CONTEXT_OVERHEAD_TOKENS
        c1 = 0
        for i in range(100, len(ct)):          # held out from the reranker fit
            ex, _ = sel.select(cq_v[i], axis=ca[i], stages=stages, band=band)
            c1 += int(bool(ex) and ex[0] == ct[i])
        tpq = toks / len(targets)
        rung[label] = {
            "recall": hits / len(targets),
            "tokens": tpq,
            "routing_top1": c1 / (len(ct) - 100),
            "joules_per_query": round(tpq * jpt, 6) if jpt else None,
        }
        print(f"E5 {label:11s} recall={rung[label]['recall']*100:5.1f}%  "
              f"tok={tpq:6.0f}  routing={rung[label]['routing_top1']*100:5.1f}%  "
              f"J/query={rung[label]['joules_per_query']}")
    out["e5_eats_natural"] = {"prefill_j_per_token": jpt, "rungs": rung}

    (GPU / "followups.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {GPU / 'followups.json'}")


if __name__ == "__main__":
    main()
