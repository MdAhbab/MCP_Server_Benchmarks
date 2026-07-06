"""
Graph-Connected Hierarchical Discovery (GHD) Benchmark
for MCP Tools and A2A Agent Capabilities.

Proposed method (GHD):
  1. Every discoverable entity (MCP tool or A2A agent capability card) is
     embedded as a multi-feature vector: a semantic text embedding (TF-IDF ->
     truncated SVD) concatenated with standardised operational features
     (log latency, energy score, log popularity, entity kind).
  2. Entities are grouped with density-based hierarchical clustering
     (HDBSCAN; Campello et al., 2013). Clusters are discovered from the data,
     not from vendor-defined server or category boundaries, so tools with
     overlapping semantics from different servers land in the same cluster.
  3. Each cluster is compressed into a ~30-token summary (top terms, member
     count, dominant kind, mean latency/energy). Only summaries live in the
     agent's context permanently.
  4. Clusters are connected into a graph via top-2 centroid-similarity edges.
     Discovery is two-stage: rank summaries, expand the best cluster plus its
     top graph neighbour, and inject only the top member definitions.
  5. Synchronisation: newly registered entities are inserted incrementally
     (nearest-centroid assignment with a similarity threshold; low-similarity
     entities open a pending singleton cluster). Full re-clustering is the
     periodic repair action. Both costs are measured.

Baselines:
  - FLAT: all entity definitions in context (status quo context stuffing).
  - RET-5: top-5 semantic retrieval per query (RAG-MCP style flat retrieval).
  - RET-B: budget-matched retrieval (top-K with K chosen so the token budget
    matches GHD), the fair equal-budget comparison.
  - GHD-NG: GHD without graph edges (ablation for the lateral links).

Metrics: context tokens per query, recall of the target entity within the
exposed set, estimated inference energy per million queries (0.1 mWh/token,
consistent with token_benchmark.py), and synchronisation cost/quality.

All randomness is seeded (seeds 42..46, five repetitions; mean +/- std).
Semantic embeddings are deterministic (TF-IDF/SVD; no external model calls).
"""

import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import HDBSCAN

SEEDS = [42, 43, 44, 45, 46]
N_TOOLS = 1000
N_AGENT_CARDS = 120
N_QUERIES = 300
N_CONSTRAINT_QUERIES = 200    # operational-constraint routing queries (Exp. B)
N_NEW_ENTITIES = 100          # simulated late registration batch
SVD_DIM = 48
OPERATIONAL_WEIGHT = 0.05     # operational features inform clustering but semantics dominate
CONSTRAINT_MARGIN = 0.10      # text-similarity band treated as functionally equivalent
MIN_CLUSTER_SIZE = 6
MIN_SAMPLES = 3
EXPAND_TOP_CLUSTERS = 3       # primary clusters routed to per query
EXPAND_MEMBER_BUDGET = 22     # max member definitions injected across expanded region
GRAPH_NEIGHBOURS = 2          # lateral edges kept per cluster
SYNC_SIM_THRESHOLD = 0.25     # below this, an incoming entity opens a pending cluster
ENERGY_WH_PER_TOKEN = 0.0001  # same assumption as token_benchmark.py
CONTEXT_OVERHEAD_TOKENS = 40  # instruction overhead added by every policy

# ---------------------------------------------------------------------------
# Synthetic but realistic corpus: 12 domains, several servers per domain, so
# semantically overlapping tools exist across servers (the real-world mess).
# ---------------------------------------------------------------------------

DOMAINS = {
    "version_control": {
        "servers": ["github", "gitlab", "bitbucket"],
        "nouns": ["repository", "branch", "commit", "pull request", "merge conflict",
                  "diff", "release tag", "issue", "code review", "workflow run"],
        "latency_ms": 180, "energy": 0.35, "compute": "io",
    },
    "messaging": {
        "servers": ["slack", "teams", "discord"],
        "nouns": ["message", "channel", "thread", "reaction", "user status",
                  "notification", "direct message", "mention", "emoji", "workspace"],
        "latency_ms": 120, "energy": 0.25, "compute": "io",
    },
    "database": {
        "servers": ["postgres", "mysql", "mongodb"],
        "nouns": ["query", "table", "row", "schema", "index", "transaction",
                  "stored procedure", "view", "backup", "connection pool"],
        "latency_ms": 90, "energy": 0.55, "compute": "cpu",
    },
    "files": {
        "servers": ["gdrive", "dropbox", "localfs"],
        "nouns": ["file", "directory", "upload", "download", "permission",
                  "shared link", "version history", "archive", "metadata", "quota"],
        "latency_ms": 150, "energy": 0.30, "compute": "io",
    },
    "email": {
        "servers": ["gmail", "outlook"],
        "nouns": ["email", "inbox", "draft", "attachment", "recipient",
                  "folder", "label", "signature", "filter rule", "calendar invite"],
        "latency_ms": 200, "energy": 0.25, "compute": "io",
    },
    "calendar": {
        "servers": ["gcal", "outlook_cal"],
        "nouns": ["event", "meeting", "reminder", "availability", "invitation",
                  "recurring schedule", "time zone", "room booking", "agenda", "attendee"],
        "latency_ms": 130, "energy": 0.20, "compute": "io",
    },
    "crm": {
        "servers": ["salesforce", "hubspot"],
        "nouns": ["contact", "lead", "deal", "pipeline stage", "account",
                  "opportunity", "activity log", "quote", "campaign", "territory"],
        "latency_ms": 250, "energy": 0.40, "compute": "io",
    },
    "payments": {
        "servers": ["stripe", "paypal"],
        "nouns": ["invoice", "payment", "refund", "subscription", "charge",
                  "payout", "dispute", "customer balance", "tax rate", "coupon"],
        "latency_ms": 300, "energy": 0.45, "compute": "io",
    },
    "cloud_infra": {
        "servers": ["aws", "azure", "gcp"],
        "nouns": ["instance", "container", "deployment", "cluster", "volume",
                  "autoscaling group", "load balancer", "security group", "snapshot", "region"],
        "latency_ms": 400, "energy": 0.80, "compute": "cpu",
    },
    "monitoring": {
        "servers": ["datadog", "grafana", "prometheus"],
        "nouns": ["alert", "metric", "dashboard", "log stream", "incident",
                  "trace span", "service map", "anomaly", "uptime check", "error rate"],
        "latency_ms": 110, "energy": 0.50, "compute": "cpu",
    },
    "web_search": {
        "servers": ["websearch", "crawler"],
        "nouns": ["search result", "web page", "crawl job", "snippet",
                  "ranking", "url", "sitemap", "cached copy", "news article", "image result"],
        "latency_ms": 350, "energy": 0.60, "compute": "io",
    },
    "data_analysis": {
        "servers": ["notebooks", "warehouse"],
        "nouns": ["dataset", "chart", "aggregation", "statistic", "report",
                  "forecast", "pivot table", "correlation", "outlier", "feature column"],
        "latency_ms": 500, "energy": 0.90, "compute": "cpu",
    },
}

VERBS = ["list", "get", "create", "update", "delete", "search", "send", "analyze"]

AGENT_ROLES = [
    ("code_review_agent", "version_control", "reviews code changes and comments on pull requests"),
    ("triage_agent", "monitoring", "triages incidents and correlates alerts with traces"),
    ("sales_assistant_agent", "crm", "qualifies leads and updates deal pipeline stages"),
    ("billing_agent", "payments", "reconciles invoices and processes refund requests"),
    ("scheduler_agent", "calendar", "negotiates meeting times and books rooms"),
    ("data_analyst_agent", "data_analysis", "builds reports and explains statistics and forecasts"),
    ("devops_agent", "cloud_infra", "manages deployments and scales clusters"),
    ("research_agent", "web_search", "searches the web and summarises pages"),
    ("inbox_agent", "email", "drafts replies and organises email folders"),
    ("librarian_agent", "files", "organises directories and manages shared links"),
    ("dba_agent", "database", "optimises queries and manages schemas and indexes"),
    ("comms_agent", "messaging", "posts updates to channels and manages notifications"),
]

FILLER = ["for the current workspace", "with pagination support", "using the default account",
          "respecting rate limits", "in the selected project", "with optional filters",
          "and return structured output", "for downstream automation"]


def make_tool(rng: random.Random, domain: str, server: str, idx: int) -> dict:
    spec = DOMAINS[domain]
    verb = rng.choice(VERBS)
    noun = rng.choice(spec["nouns"])
    filler = rng.choice(FILLER)
    name = f"{server}_{verb}_{noun.replace(' ', '_')}_{idx}"
    description = (f"{verb.capitalize()} {noun} on {server}. "
                   f"This tool lets an agent {verb} the {noun} {filler}.")
    n_params = rng.randint(2, 5)
    schema = {
        "type": "object",
        "properties": {f"param_{j}": {"type": rng.choice(["string", "integer", "boolean"]),
                                      "description": f"The {noun} {['id','name','filter','limit','flag'][j % 5]}"}
                       for j in range(n_params)},
        "required": ["param_0"],
    }
    latency = rng.lognormvariate(math.log(spec["latency_ms"]), 0.35)
    return {
        "kind": "mcp_tool", "domain": domain, "server": server,
        "name": name, "description": description, "inputSchema": schema,
        "latency_ms": latency, "energy_score": spec["energy"],
        "verb": verb, "noun": noun,
    }


def make_agent_card(rng: random.Random, role: str, domain: str, blurb: str, idx: int) -> dict:
    spec = DOMAINS[domain]
    nouns = rng.sample(spec["nouns"], k=3)
    name = f"{role}_{idx}"
    description = (f"A2A agent capability: {role.replace('_', ' ')}. "
                   f"This agent {blurb}. It works with {nouns[0]}, {nouns[1]}, and {nouns[2]}.")
    latency = rng.lognormvariate(math.log(spec["latency_ms"] * 4), 0.4)  # delegation is slower
    return {
        "kind": "a2a_capability", "domain": domain, "server": role,
        "name": name, "description": description,
        "inputSchema": {"type": "object", "properties": {"task": {"type": "string",
                        "description": "Natural-language task delegated to this agent"}},
                        "required": ["task"]},
        "latency_ms": latency, "energy_score": min(1.0, spec["energy"] + 0.2),
        "verb": "delegate", "noun": rng.choice(nouns),
    }


def build_corpus(seed: int, n_tools: int = N_TOOLS, n_agents: int = N_AGENT_CARDS) -> list:
    rng = random.Random(seed)
    entities = []
    domains = list(DOMAINS)
    for i in range(n_tools):
        domain = domains[i % len(domains)]
        server = rng.choice(DOMAINS[domain]["servers"])
        entities.append(make_tool(rng, domain, server, i))
    for i in range(n_agents):
        role, domain, blurb = AGENT_ROLES[i % len(AGENT_ROLES)]
        entities.append(make_agent_card(rng, role, domain, blurb, i))
    # Popularity: Zipf-like, so query sampling mirrors real usage concentration.
    ranks = list(range(1, len(entities) + 1))
    rng.shuffle(ranks)
    for ent, r in zip(entities, ranks):
        ent["popularity"] = 1.0 / r
    for ent in entities:
        ent["def_text"] = json.dumps({"name": ent["name"], "description": ent["description"],
                                      "inputSchema": ent["inputSchema"]})
        ent["def_tokens"] = count_tokens(ent["def_text"])
    return entities


# ---------------------------------------------------------------------------
# Embedding pipeline (deterministic, no external model)
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """TF-IDF -> truncated SVD text embedding plus standardised operational
    features. The text block is L2-normalised so cosine similarity in the
    text subspace is a dot product."""

    def __init__(self, svd_dim: int = SVD_DIM, op_weight: float = OPERATIONAL_WEIGHT):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                          min_df=2, stop_words="english")
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=0)
        self.scaler = StandardScaler()
        self.op_weight = op_weight

    @staticmethod
    def _texts(entities):
        return [f"{e['name'].replace('_', ' ')}. {e['description']}" for e in entities]

    @staticmethod
    def _ops(entities):
        return np.array([[math.log(e["latency_ms"]), e["energy_score"],
                          math.log(e["popularity"]), 1.0 if e["kind"] == "a2a_capability" else 0.0]
                         for e in entities])

    def fit_transform(self, entities):
        tfidf = self.vectorizer.fit_transform(self._texts(entities))
        text_vec = normalize(self.svd.fit_transform(tfidf))
        op_vec = self.scaler.fit_transform(self._ops(entities)) * self.op_weight
        return np.hstack([text_vec, op_vec]), text_vec

    def transform(self, entities):
        tfidf = self.vectorizer.transform(self._texts(entities))
        text_vec = normalize(self.svd.transform(tfidf))
        op_vec = self.scaler.transform(self._ops(entities)) * self.op_weight
        return np.hstack([text_vec, op_vec]), text_vec

    def embed_queries(self, queries):
        tfidf = self.vectorizer.transform(queries)
        return normalize(self.svd.transform(tfidf))


# ---------------------------------------------------------------------------
# GHD hierarchy: HDBSCAN clusters + summaries + lateral graph edges
# ---------------------------------------------------------------------------

class GHDHierarchy:
    def __init__(self, entities, features, text_vecs):
        self.entities = entities
        self.features = features
        self.text_vecs = text_vecs
        t0 = time.perf_counter()
        clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
        labels = clusterer.fit_predict(features)
        self.build_time_s = time.perf_counter() - t0
        self.noise_grafted = int(np.sum(labels == -1))
        # Graft noise points onto the nearest discovered cluster.
        cluster_ids = sorted(set(labels) - {-1})
        centroids = {c: text_vecs[labels == c].mean(axis=0) for c in cluster_ids}
        cent_mat = normalize(np.vstack([centroids[c] for c in cluster_ids]))
        for i in np.where(labels == -1)[0]:
            sims = cent_mat @ (text_vecs[i] / (np.linalg.norm(text_vecs[i]) + 1e-12))
            labels[i] = cluster_ids[int(np.argmax(sims))]
        self.labels = labels
        self.cluster_ids = cluster_ids
        self.centroids = normalize(np.vstack(
            [text_vecs[labels == c].mean(axis=0) for c in cluster_ids]))
        self.members = {c: np.where(labels == c)[0] for c in cluster_ids}
        # Lateral graph edges: top-N most similar other clusters.
        sim = self.centroids @ self.centroids.T
        np.fill_diagonal(sim, -1.0)
        self.neighbours = {c: [cluster_ids[j] for j in np.argsort(-sim[k])[:GRAPH_NEIGHBOURS]]
                           for k, c in enumerate(cluster_ids)}
        self.summaries = {c: self._summarise(c) for c in cluster_ids}
        self.summary_tokens = sum(count_tokens(s) for s in self.summaries.values())

    def _summarise(self, c):
        idx = self.members[c]
        docs = [f"{self.entities[i]['verb']} {self.entities[i]['noun']}" for i in idx]
        term_counts = {}
        for d in docs:
            for w in d.split():
                term_counts[w] = term_counts.get(w, 0) + 1
        top_terms = [w for w, _ in sorted(term_counts.items(), key=lambda x: -x[1])[:6]]
        kinds = [self.entities[i]["kind"] for i in idx]
        dominant = "agents" if kinds.count("a2a_capability") > len(kinds) / 2 else "tools"
        mean_lat = statistics.mean(self.entities[i]["latency_ms"] for i in idx)
        mean_en = statistics.mean(self.entities[i]["energy_score"] for i in idx)
        return (f"[cluster {c}] {len(idx)} {dominant}: {', '.join(top_terms)}. "
                f"avg latency {mean_lat:.0f} ms, energy score {mean_en:.2f}.")

    def discover(self, q_vec, use_graph=True, op_pref=None,
                 member_budget=EXPAND_MEMBER_BUDGET):
        """Two-stage hierarchical discovery.

        Stage 1 (coarse): rank clusters by centroid similarity, take the top
        EXPAND_TOP_CLUSTERS. When use_graph, also add the top-1 graph neighbour
        of the best cluster, which catches queries that fall between domains.
        Stage 2 (fine): pool the members of the expanded clusters, rank them by
        query similarity (plus an operational-preference bonus when op_pref is
        given), and inject the top member_budget definitions.

        op_pref: None, or ('latency'|'energy', weight). A positive weight
        favours entities that are cheaper on that operational axis. Only GHD can
        honour this, because operational features are part of its index.
        Returns (exposed indices, context tokens).
        """
        cl_sims = self.centroids @ q_vec
        ranked = list(np.argsort(-cl_sims))
        chosen = ranked[:EXPAND_TOP_CLUSTERS]
        expand = [self.cluster_ids[k] for k in chosen]
        if use_graph:
            for nb in self.neighbours[self.cluster_ids[ranked[0]]]:
                if nb not in expand:
                    expand.append(nb)
                    break
        pool = np.concatenate([self.members[c] for c in expand])
        scores = self.text_vecs[pool] @ q_vec
        if op_pref is None:
            order = np.argsort(-scores)[:member_budget]
        else:
            # Two-stage operational selection. First identify the functional
            # matches: candidates within CONSTRAINT_MARGIN of the best text
            # score. Then order those by the requested operational axis
            # (cheaper first), so the operationally-best functional match ranks
            # #1. Remaining budget is filled by text score. Pure text retrieval
            # cannot do this because it does not index operational features.
            axis, _ = op_pref
            if axis == "latency":
                op = np.array([self.entities[i]["latency_ms"] for i in pool])
            else:
                op = np.array([self.entities[i]["energy_score"] for i in pool])
            best = scores.max()
            functional = np.where(scores >= best - CONSTRAINT_MARGIN)[0]
            functional = functional[np.argsort(op[functional])]  # cheapest first
            rest = [j for j in np.argsort(-scores) if j not in set(functional)]
            order = list(functional) + rest
            order = order[:member_budget]
        exposed = [int(pool[o]) for o in order]
        tokens = (self.summary_tokens + CONTEXT_OVERHEAD_TOKENS +
                  sum(self.entities[i]["def_tokens"] for i in exposed))
        return exposed, tokens

    def incremental_insert(self, new_text_vecs):
        """Nearest-centroid assignment; low-similarity entities open pending
        singleton clusters. Returns (assignments, n_pending, elapsed_s)."""
        t0 = time.perf_counter()
        assignments, pending = [], 0
        for v in normalize(new_text_vecs):
            sims = self.centroids @ v
            j = int(np.argmax(sims))
            if sims[j] >= SYNC_SIM_THRESHOLD:
                assignments.append(self.cluster_ids[j])
            else:
                assignments.append(-1)
                pending += 1
        return assignments, pending, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Query generation and policy evaluation
# ---------------------------------------------------------------------------

def make_queries(rng: random.Random, entities, n_queries: int):
    weights = [e["popularity"] for e in entities]
    targets = rng.choices(range(len(entities)), weights=weights, k=n_queries)
    queries = []
    for t in targets:
        e = entities[t]
        extra = rng.choice(DOMAINS[e["domain"]]["nouns"])
        noise = rng.choice(FILLER)
        if e["kind"] == "a2a_capability":
            q = f"delegate this task: {e['description'].split(':')[1].split('.')[0]} regarding {extra}"
        else:
            q = f"please {e['verb']} the {e['noun']} related to {extra} {noise}"
        queries.append(q)
    return targets, queries


def build_constraint_groups(entities):
    """Group near-duplicate entities that share (domain, verb, noun) but live on
    different servers with different latency/energy. These are the cases where an
    agent must choose the operationally cheapest option among functionally
    equivalent tools."""
    groups = {}
    for i, e in enumerate(entities):
        if e["kind"] != "mcp_tool":
            continue
        key = (e["domain"], e["verb"], e["noun"])
        groups.setdefault(key, []).append(i)
    return {k: v for k, v in groups.items() if len(v) >= 3}


def make_constraint_queries(rng, entities, groups, n):
    """Each query has a semantic intent plus an operational preference (minimise
    latency or energy). Target = the group member that is cheapest on that axis."""
    keys = list(groups)
    queries, targets, prefs = [], [], []
    for _ in range(n):
        key = rng.choice(keys)
        members = groups[key]
        domain, verb, noun = key
        axis = rng.choice(["latency", "energy"])
        if axis == "latency":
            target = min(members, key=lambda i: entities[i]["latency_ms"])
            phrase = "with the lowest latency"
        else:
            target = min(members, key=lambda i: entities[i]["energy_score"])
            phrase = "that is the most energy efficient"
        extra = rng.choice(DOMAINS[domain]["nouns"])
        q = f"please {verb} the {noun} related to {extra}, choosing the option {phrase}"
        queries.append(q)
        targets.append(target)
        prefs.append(axis)
    return targets, queries, prefs


def evaluate_rep(seed: int) -> dict:
    rng = random.Random(seed + 1000)
    entities = build_corpus(seed)
    pipeline = EmbeddingPipeline()
    features, text_vecs = pipeline.fit_transform(entities)
    hierarchy = GHDHierarchy(entities, features, text_vecs)

    targets, queries = make_queries(rng, entities, N_QUERIES)
    q_vecs = pipeline.embed_queries(queries)

    flat_tokens = sum(e["def_tokens"] for e in entities) + CONTEXT_OVERHEAD_TOKENS
    results = {p: {"tokens": [], "hits": 0} for p in ["FLAT", "RET-5", "RET-B", "GHD", "GHD-NG"]}

    # --- Experiment A: semantic discovery -------------------------------
    # Compute GHD's realised token budget first, then match RET-B to it so the
    # retrieval baseline and GHD see the same context budget (fair comparison).
    ghd_budget_probe = [hierarchy.discover(q_vecs[i])[1] for i in range(min(60, N_QUERIES))]
    mean_ghd_budget = statistics.mean(ghd_budget_probe)
    mean_def_tokens = statistics.mean(e["def_tokens"] for e in entities)
    k_budget = max(5, int((mean_ghd_budget - CONTEXT_OVERHEAD_TOKENS) / mean_def_tokens))

    for i, t in enumerate(targets):
        qv = q_vecs[i]
        sims = text_vecs @ qv

        results["FLAT"]["tokens"].append(flat_tokens)
        results["FLAT"]["hits"] += 1  # every definition is already in context

        top5 = np.argsort(-sims)[:5]
        results["RET-5"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in top5))
        results["RET-5"]["hits"] += int(t in top5)

        topb = np.argsort(-sims)[:k_budget]
        results["RET-B"]["tokens"].append(
            CONTEXT_OVERHEAD_TOKENS + sum(entities[j]["def_tokens"] for j in topb))
        results["RET-B"]["hits"] += int(t in topb)

        exposed, tokens = hierarchy.discover(qv, use_graph=True)
        results["GHD"]["tokens"].append(tokens)
        results["GHD"]["hits"] += int(t in exposed)

        exposed_ng, tokens_ng = hierarchy.discover(qv, use_graph=False)
        results["GHD-NG"]["tokens"].append(tokens_ng)
        results["GHD-NG"]["hits"] += int(t in exposed_ng)

    # --- Experiment B: operational-constraint routing -------------------
    # Among functionally equivalent tools, pick the operationally cheapest.
    # Semantic retrieval ranks by text only and cannot honour the preference;
    # GHD re-ranks the exposed pool by the operational feature it indexes.
    groups = build_constraint_groups(entities)
    con_targets, con_queries, con_prefs = make_constraint_queries(
        rng, entities, groups, N_CONSTRAINT_QUERIES)
    con_q_vecs = pipeline.embed_queries(con_queries)
    con = {"RET_top1": 0, "RET_cover": 0, "GHD_top1": 0, "GHD_cover": 0, "n": N_CONSTRAINT_QUERIES}
    for i, t in enumerate(con_targets):
        qv = con_q_vecs[i]
        axis = con_prefs[i]
        sims = text_vecs @ qv
        # Semantic retrieval (RAG-MCP style): rank by text similarity only.
        topb = list(np.argsort(-sims)[:k_budget])
        con["RET_cover"] += int(t in topb)
        con["RET_top1"] += int(len(topb) > 0 and topb[0] == t)
        # GHD with an explicit operational preference passed by the agent.
        exposed, _ = hierarchy.discover(qv, use_graph=True, op_pref=(axis, 1.0))
        con["GHD_cover"] += int(t in exposed)
        con["GHD_top1"] += int(len(exposed) > 0 and exposed[0] == t)

    # --- Synchronisation experiment -------------------------------------
    new_entities = []
    rng2 = random.Random(seed + 2000)
    for i in range(N_NEW_ENTITIES):
        domain = rng2.choice(list(DOMAINS))
        server = rng2.choice(DOMAINS[domain]["servers"])
        new_entities.append(make_tool(rng2, domain, server, 10_000 + i))
    for e in new_entities:
        e["popularity"] = 1.0 / rng2.randint(1, len(entities))
        e["def_text"] = json.dumps({"name": e["name"], "description": e["description"],
                                    "inputSchema": e["inputSchema"]})
        e["def_tokens"] = count_tokens(e["def_text"])
    _, new_text_vecs = pipeline.transform(new_entities)
    _, n_pending, inc_time = hierarchy.incremental_insert(new_text_vecs)

    t0 = time.perf_counter()
    all_entities = entities + new_entities
    pipeline_full = EmbeddingPipeline()
    features_full, text_full = pipeline_full.fit_transform(all_entities)
    GHDHierarchy(all_entities, features_full, text_full)
    full_time = time.perf_counter() - t0

    # Recall on queries that target the newly registered tools (incremental).
    new_targets_local, new_queries = make_queries(rng2, new_entities, 100)
    nq_vecs = pipeline.embed_queries(new_queries)
    ext_text_vecs = normalize(new_text_vecs)
    inc_hits = 0
    for qi, tl in enumerate(new_targets_local):
        qv = nq_vecs[qi]
        sims = hierarchy.centroids @ qv
        best = hierarchy.cluster_ids[int(np.argmax(sims))]
        expand = [best] + hierarchy.neighbours[best][:1]
        assign, _, _ = hierarchy.incremental_insert(ext_text_vecs[tl:tl + 1])
        inc_hits += int(assign[0] in expand)

    n_entities = len(entities)
    summary = {
        "seed": seed,
        "n_entities": n_entities,
        "n_clusters": len(hierarchy.cluster_ids),
        "noise_grafted": hierarchy.noise_grafted,
        "summary_tokens": hierarchy.summary_tokens,
        "build_time_s": hierarchy.build_time_s,
        "k_budget_matched": k_budget,
        "flat_tokens": flat_tokens,
        "policies": {},
        "sync": {
            "incremental_insert_s": inc_time,
            "incremental_pending": n_pending,
            "full_recluster_s": full_time,
            "incremental_recall_new": inc_hits / 100,
        },
        "constraint": {
            "n": con["n"],
            "ret_top1_recall": con["RET_top1"] / con["n"],
            "ret_coverage": con["RET_cover"] / con["n"],
            "ghd_top1_recall": con["GHD_top1"] / con["n"],
            "ghd_coverage": con["GHD_cover"] / con["n"],
        },
    }
    for p, r in results.items():
        mean_tok = statistics.mean(r["tokens"])
        summary["policies"][p] = {
            "mean_tokens": mean_tok,
            "std_tokens": statistics.stdev(r["tokens"]) if len(set(r["tokens"])) > 1 else 0.0,
            "recall": r["hits"] / N_QUERIES,
            "token_reduction_vs_flat_percent": (1 - mean_tok / flat_tokens) * 100,
            "energy_wh_per_million_queries": mean_tok * ENERGY_WH_PER_TOKEN * 1_000_000,
        }
    return summary


def aggregate(reps):
    policies = list(reps[0]["policies"])
    agg = {"repetitions": len(reps), "seeds": [r["seed"] for r in reps],
           "n_entities": reps[0]["n_entities"],
           "n_clusters_mean": statistics.mean(r["n_clusters"] for r in reps),
           "n_clusters_std": statistics.stdev(r["n_clusters"] for r in reps),
           "summary_tokens_mean": statistics.mean(r["summary_tokens"] for r in reps),
           "build_time_s_mean": statistics.mean(r["build_time_s"] for r in reps),
           "flat_tokens_mean": statistics.mean(r["flat_tokens"] for r in reps),
           "policies": {}, "sync": {}}
    for p in policies:
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
    for key in ["incremental_insert_s", "incremental_pending", "full_recluster_s",
                "incremental_recall_new"]:
        vals = [r["sync"][key] for r in reps]
        agg["sync"][key + "_mean"] = statistics.mean(vals)
        agg["sync"][key + "_std"] = statistics.stdev(vals)
    agg["constraint"] = {}
    for key in ["ret_top1_recall", "ret_coverage", "ghd_top1_recall", "ghd_coverage"]:
        vals = [r["constraint"][key] for r in reps]
        agg["constraint"][key + "_mean"] = statistics.mean(vals)
        agg["constraint"][key + "_std"] = statistics.stdev(vals)
    return agg


def make_figure(agg, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping figure")
        return
    order = ["FLAT", "RET-5", "RET-B", "GHD-NG", "GHD"]
    labels = ["Flat\n(all defs)", "Top-5\nretrieval", "Budget-\nmatched\nretrieval",
              "GHD\n(no graph)", "GHD\n(proposed)"]
    tokens = [agg["policies"][p]["mean_tokens"] for p in order]
    tok_std = [agg["policies"][p]["std_tokens"] for p in order]
    recalls = [agg["policies"][p]["recall_mean"] * 100 for p in order]
    rec_std = [agg["policies"][p]["recall_std"] * 100 for p in order]
    colors = ["#e74c3c", "#f39c12", "#f1c40f", "#85c1e9", "#27ae60"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel (a): context cost per query
    bars = ax1.bar(labels, tokens, yerr=tok_std, capsize=4, color=colors, ecolor="gray")
    ax1.set_yscale("log")
    ax1.set_ylabel("Context tokens per query (log scale)", fontsize=11)
    ax1.set_title("(a) Context Cost per Query", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", labelsize=8)
    for b, t in zip(bars, tokens):
        ax1.text(b.get_x() + b.get_width() / 2, t * 1.18, f"{t:,.0f}",
                 ha="center", va="bottom", fontsize=8)

    # Panel (b): semantic discovery recall
    bars2 = ax2.bar(labels, recalls, yerr=rec_std, capsize=4, color=colors, ecolor="gray")
    ax2.set_ylabel("Recall of target entity (%)", fontsize=11)
    ax2.set_ylim(0, 108)
    ax2.set_title("(b) Semantic Discovery Recall", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", labelsize=8)
    for b, r in zip(bars2, recalls):
        ax2.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.1f}",
                 ha="center", va="bottom", fontsize=8)

    # Panel (c): operational-constraint routing (semantic retrieval vs GHD)
    c = agg["constraint"]
    con_labels = ["Semantic\nretrieval", "GHD\n(proposed)"]
    con_vals = [c["ret_top1_recall_mean"] * 100, c["ghd_top1_recall_mean"] * 100]
    con_std = [c["ret_top1_recall_std"] * 100, c["ghd_top1_recall_std"] * 100]
    bars3 = ax3.bar(con_labels, con_vals, yerr=con_std, capsize=4,
                    color=["#f1c40f", "#27ae60"], ecolor="gray")
    ax3.set_ylabel("Constraint-correct top-1 selection (%)", fontsize=11)
    ax3.set_ylim(0, max(con_vals) * 1.5)
    ax3.set_title("(c) Operational-Constraint Routing", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars3, con_vals):
        ax3.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "hierarchical_discovery.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "hierarchical_discovery.pdf", bbox_inches="tight")
    plt.close()
    print("✓ Generated hierarchical_discovery.png/pdf")


def make_latex_table(agg, out_path: Path):
    p = agg["policies"]
    rows = [("Flat (all definitions)", "FLAT"), ("Top-5 retrieval", "RET-5"),
            ("Budget-matched retrieval", "RET-B"), ("GHD without graph edges", "GHD-NG"),
            ("GHD (proposed)", "GHD")]
    latex = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Discovery Policies over "
             f"{agg['n_entities']:,} Entities (Mean $\\pm$ SD, Five Seeded Runs)}}",
             "\\label{tab:ghd_benchmark}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "\\textbf{Policy} & \\textbf{Tokens/query} & \\textbf{Recall (\\%)} & \\textbf{Wh/10\\textsuperscript{6} queries} \\\\",
             "\\midrule"]
    for label, key in rows:
        d = p[key]
        latex.append(f"{label} & {d['mean_tokens']:,.0f} $\\pm$ {d['std_tokens']:,.0f} & "
                     f"{d['recall_mean']*100:.1f} $\\pm$ {d['recall_std']*100:.1f} & "
                     f"{d['energy_wh_per_million_queries']:,.0f} \\\\")
    latex += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(latex), encoding="utf-8")
    print(f"✓ LaTeX table written to {out_path}")


def run_benchmark():
    print("=" * 70)
    print("Graph-Connected Hierarchical Discovery (GHD) Benchmark")
    print("=" * 70)
    reps = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        rep = evaluate_rep(seed)
        reps.append(rep)
        print(f"seed {seed}: {rep['n_clusters']} clusters, "
              f"GHD tokens {rep['policies']['GHD']['mean_tokens']:.0f}, "
              f"GHD recall {rep['policies']['GHD']['recall']*100:.1f}%, "
              f"({time.perf_counter()-t0:.1f}s)")

    agg = aggregate(reps)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "entities": f"{N_TOOLS} MCP tools + {N_AGENT_CARDS} A2A capability cards, "
                        "12 domains, 2-3 servers/domain (overlapping semantics)",
            "embedding": f"TF-IDF (1-2 grams) -> SVD d={SVD_DIM}, L2-normalised, "
                         f"+ operational features (log latency, energy, log popularity, kind) "
                         f"x{OPERATIONAL_WEIGHT}",
            "clustering": f"HDBSCAN(min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}); "
                          "noise grafted to nearest centroid",
            "graph": f"top-{GRAPH_NEIGHBOURS} centroid-similarity edges per cluster; "
                     f"discovery expands top-{EXPAND_TOP_CLUSTERS} clusters + top-1 "
                     f"graph neighbour, then ranks up to {EXPAND_MEMBER_BUDGET} pooled members",
            "queries": f"{N_QUERIES} popularity-weighted synthetic intents per repetition",
            "seeds": SEEDS,
            "energy_assumption_wh_per_token": ENERGY_WH_PER_TOKEN,
            "note": "Recall measures whether the target entity is exposed to the model; "
                    "no LLM controller is in the loop (see main paper measurement card).",
        },
        "aggregate": agg,
        "repetitions": reps,
    }
    out = Path("results/hierarchical_discovery_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {out}")

    make_figure(agg, Path("results/figures"))
    make_latex_table(agg, Path("results/latex/ghd_table.tex"))

    print("\nSUMMARY (mean over 5 seeds)")
    for pol, d in agg["policies"].items():
        print(f"  {pol:7s} tokens={d['mean_tokens']:9,.0f}  recall={d['recall_mean']*100:5.1f}%  "
              f"reduction vs flat={d['token_reduction_vs_flat_percent']:5.1f}%")
    c = agg["constraint"]
    print(f"  constraint routing (top-1 recall): semantic retrieval "
          f"{c['ret_top1_recall_mean']*100:.1f}% vs GHD {c['ghd_top1_recall_mean']*100:.1f}% "
          f"(coverage {c['ret_coverage_mean']*100:.1f}% vs {c['ghd_coverage_mean']*100:.1f}%)")
    s = agg["sync"]
    print(f"  sync: incremental insert {s['incremental_insert_s_mean']*1000:.1f} ms "
          f"vs full recluster {s['full_recluster_s_mean']:.2f} s "
          f"({s['full_recluster_s_mean']/max(s['incremental_insert_s_mean'],1e-9):.0f}x); "
          f"incremental recall on new tools {s['incremental_recall_new_mean']*100:.1f}%")
    return results


if __name__ == "__main__":
    run_benchmark()
