"""
EATS: Energy-Aware Tool Selection.

A selection layer that sits on top of flat semantic retrieval. Our discovery
experiments found that budget-matched flat retrieval is the strongest policy on
the public MCP registry, and that what moves energy-aware tool choice is
indexing operational metadata rather than imposing an index structure. EATS
therefore keeps flat retrieval and improves the step that follows it.

The layer is a stack of stages. Each is independently switchable, so the
ablation is a ladder rather than a single system:

  R  Retrieve            budget-matched top-k semantic retrieval (the baseline)
  A  Adaptive band       parameter-free equivalence band from the similarity curve
  P  Pareto filter       drop candidates dominated on (relevance, cost)
  C  Cost-aware rank     order by sim - lambda * normalised cost
  B  Adaptive budget     spend fewer definitions when the query is unambiguous
  L  Learned rerank      logistic model over semantic and operational features

Stages A and B are the substantive contributions.

A replaces a hand-tuned similarity margin. A fixed margin is not portable across
encoders: a margin calibrated on term-frequency cosines mis-fires on a neural
encoder, whose similarity range is compressed. The adaptive band reads the band
off each query's own similarity curve instead.

B is the energy lever. Retrieval normally injects a fixed number of tool
definitions per query. Many queries have one obvious answer and do not need the
full budget. Sizing the budget by the ambiguity of the similarity distribution
cuts tokens, and therefore prefill energy, with no change to the ranking rule.

All stages are deterministic given their inputs.
"""

from __future__ import annotations

import numpy as np

STAGE_ORDER = "RAPCBL"


# ---------------------------------------------------------------------------
# Stage A: adaptive equivalence band
# ---------------------------------------------------------------------------

def relative_band(sims_desc: np.ndarray, alpha: float = 0.25, min_keep: int = 1) -> int:
    """Band width as a fraction of the query's own similarity spread.

    An absolute margin does not transfer between encoders: term-frequency
    cosines spread across most of [0, 1] while a neural encoder compresses them
    into a narrow high band, so the same numeric margin selects a sensible set
    in one space and almost nothing in the other. Scaling the margin by the
    observed spread of this query's candidates removes that dependence, at the
    cost of one interpretable parameter, alpha, which is a proportion rather
    than a similarity value.
    """
    n = len(sims_desc)
    if n <= 1:
        return n
    hi, lo = float(sims_desc[0]), float(sims_desc[-1])
    spread = hi - lo
    if spread <= 1e-9:
        return n
    cut = hi - alpha * spread
    return max(min_keep, int((sims_desc >= cut).sum()))


def adaptive_band(sims_desc: np.ndarray, max_frac: float = 0.5, min_keep: int = 1) -> int:
    """Return how many leading candidates form the functional-equivalence band.

    Candidates are assumed sorted by descending similarity. The band ends at the
    largest drop in the curve: the point where the next candidate is materially
    less similar than the current one. This is the one-dimensional analogue of a
    knee, and it needs no threshold, so it transfers across encoders whose
    similarity scales differ.

    max_frac bounds the search to the leading half of the pool, which prevents a
    late gap in a long tail of irrelevant candidates from being mistaken for the
    band edge. min_keep guarantees a non-empty band.
    """
    n = len(sims_desc)
    if n <= 1:
        return n
    horizon = max(min_keep + 1, int(np.ceil(n * max_frac)))
    horizon = min(horizon, n)
    gaps = sims_desc[: horizon - 1] - sims_desc[1:horizon]
    if gaps.size == 0:
        return min_keep
    return int(np.argmax(gaps)) + 1


# ---------------------------------------------------------------------------
# Stage P: Pareto filter over (relevance up, cost down)
# ---------------------------------------------------------------------------

def pareto_front(sims: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Indices of candidates not dominated on both relevance and cost.

    Candidate j dominates i when it is at least as relevant and at least as
    cheap, and strictly better on one axis. Keeping only the front removes
    options no rational selector would take, without needing a weight between
    the two axes.
    """
    order = np.argsort(-sims)
    keep, best_cost = [], np.inf
    for idx in order:
        c = cost[idx]
        if c < best_cost:
            keep.append(int(idx))
            best_cost = c
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# Stage B: adaptive token budget
# ---------------------------------------------------------------------------

def ambiguity(sims_desc: np.ndarray, top: int = 10) -> float:
    """Ambiguity of a query in [0, 1], from the shape of its similarity curve.

    A query whose best match stands well clear of the runners-up is
    unambiguous and scores near 0. A query whose leading candidates are packed
    together scores near 1. We combine the normalised top-1 margin with the
    entropy of a softmax over the leading similarities, so that both a close
    second place and a broad flat head raise the score.
    """
    s = sims_desc[:top]
    if s.size < 2:
        return 0.0
    spread = float(s[0] - s[-1])
    margin = float(s[0] - s[1]) / spread if spread > 1e-9 else 0.0
    z = s - s.max()
    p = np.exp(z / max(1e-6, spread * 0.5))
    p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return float(np.clip(0.5 * (1.0 - np.clip(margin, 0.0, 1.0)) + 0.5 * ent, 0.0, 1.0))


def adaptive_budget(sims_desc: np.ndarray, k_max: int, k_min: int = 3) -> int:
    """Definitions to inject for this query, between k_min and k_max."""
    a = ambiguity(sims_desc)
    return int(np.clip(round(k_min + (k_max - k_min) * a), k_min, k_max))


# ---------------------------------------------------------------------------
# Stage L: learned reranker
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["sim", "sim_gap_to_best", "requested_cost_z", "requested_cost_rank",
                 "latency_z", "energy_z", "log_popularity", "is_agent"]


def features(sims: np.ndarray, cand: np.ndarray, entities, axis: str | None,
             lat_mu: float, lat_sd: float, en_mu: float, en_sd: float) -> np.ndarray:
    """Feature matrix for the candidates of one query."""
    best = sims.max() if sims.size else 0.0
    key = "latency_ms" if axis == "latency" else "energy_score"
    mu, sd = (lat_mu, lat_sd) if axis == "latency" else (en_mu, en_sd)
    req = np.array([float(entities[i][key]) for i in cand])
    # Rank of each candidate by cost within this query's own candidate set. The
    # absolute z-score alone cannot express "cheapest among these", which is
    # exactly what the task asks for.
    rank = np.empty(len(req))
    rank[np.argsort(req)] = np.arange(len(req)) / max(1, len(req) - 1)
    rows = []
    for j, i in enumerate(cand):
        e = entities[i]
        rows.append([
            float(sims[j]),
            float(sims[j] - best),
            (req[j] - mu) / (sd + 1e-9),
            float(rank[j]),
            (float(e["latency_ms"]) - lat_mu) / (lat_sd + 1e-9),
            (float(e["energy_score"]) - en_mu) / (en_sd + 1e-9),
            float(np.log(max(e.get("popularity", 1e-6), 1e-6))),
            1.0 if e.get("kind") == "a2a_capability" else 0.0,
        ])
    return np.asarray(rows, dtype=np.float64)


class LearnedReranker:
    """Logistic model scoring how likely a candidate is the intended tool.

    Deliberately small. The point of the stage is to show whether a light
    supervised layer adds anything over the parameter-free rules, not to search
    model space. Trained on held-out seeds so no query it scores was seen.
    """

    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        if len(np.unique(y)) < 2:
            self.model = None
            return self
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        ).fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or X.size == 0:
            return np.zeros(len(X))
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------

class EATSSelector:
    """Energy-aware selection over a flat semantic index.

    entities carry the operational fields the layer needs: latency_ms,
    energy_score, popularity, kind, def_tokens.
    """

    def __init__(self, entities, text_vecs: np.ndarray, k_max: int,
                 lam: float = 1.0, reranker: LearnedReranker | None = None,
                 alpha: float = 0.25):
        self.entities = entities
        self.text_vecs = text_vecs
        self.k_max = int(k_max)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.reranker = reranker
        lat = np.array([e["latency_ms"] for e in entities], dtype=float)
        en = np.array([e["energy_score"] for e in entities], dtype=float)
        self.lat_mu, self.lat_sd = lat.mean(), lat.std()
        self.en_mu, self.en_sd = en.mean(), en.std()
        self._cost = {"latency": lat, "energy": en}

    def select(self, qv: np.ndarray, axis: str | None = None,
               stages: str = "RAPCBL", band: str = "adaptive",
               fixed_margin: float = 0.10) -> tuple[list[int], int]:
        """Return (ordered exposed indices, definition tokens spent).

        stages is any subset of RAPCBL; R is always applied. band selects how
        the functional-equivalence set is drawn: "adaptive" uses stage A,
        "fixed" uses a constant similarity margin, which is the rule stage A
        replaces and the reference point for its ablation.
        """
        sims_all = self.text_vecs @ qv
        order = np.argsort(-sims_all)

        # R plus optional B: how many candidates to consider and expose.
        k = self.k_max
        if "B" in stages:
            k = adaptive_budget(sims_all[order[: max(self.k_max, 10)]], self.k_max)
        cand = order[:k]
        sims = sims_all[cand]

        # A: restrict the reordering to a functional-equivalence band. Outside
        # the band, candidates are not interchangeable, so operational cost must
        # not be allowed to promote them over a better semantic match.
        if band == "fixed":
            band_n = int((sims >= sims.max() - fixed_margin).sum())
        elif band == "relative" and "A" in stages:
            band_n = relative_band(sims, self.alpha)
        elif "A" in stages:
            band_n = adaptive_band(sims)
        else:
            band_n = len(sims)
        band_n = max(1, band_n)
        band_idx = np.arange(band_n)
        tail_idx = np.arange(band_n, len(sims))

        if axis is not None and ("P" in stages or "C" in stages or "L" in stages):
            cost_all = self._cost[axis]
            bcost = cost_all[cand[band_idx]]
            bsims = sims[band_idx]

            sel = band_idx
            dropped = np.array([], dtype=int)
            if "P" in stages:
                # Drop, do not merely reorder. A candidate that is both less
                # relevant and more costly than another will never be chosen, so
                # spending definition tokens to expose it is waste. Ordering it
                # lower would achieve nothing, because stage C re-sorts anyway.
                front = pareto_front(bsims, bcost)
                if front.size:
                    keep = set(front.tolist())
                    dropped = np.array([j for j in band_idx if j not in keep], dtype=int)
                    sel = band_idx[front]

            if "C" in stages:
                s = sims[sel]
                c = cost_all[cand[sel]]
                s_n = (s - s.min()) / (np.ptp(s) + 1e-9)
                c_n = (c - c.min()) / (np.ptp(c) + 1e-9)
                sel = sel[np.argsort(-(s_n - self.lam * c_n))]

            if "L" in stages and self.reranker is not None and self.reranker.model is not None:
                X = features(sims[sel], cand[sel], self.entities, axis,
                             self.lat_mu, self.lat_sd, self.en_mu, self.en_sd)
                sel = sel[np.argsort(-self.reranker.score(X))]

            final = np.concatenate([sel, tail_idx]).astype(int)
        else:
            final = np.arange(len(sims))

        exposed = [int(cand[j]) for j in final]
        tokens = sum(self.entities[i]["def_tokens"] for i in exposed)
        return exposed, tokens


def build_training_set(selector: EATSSelector, q_vecs, targets, axes):
    """Candidate-level training data for the learned stage."""
    X, y = [], []
    for qi, qv in enumerate(q_vecs):
        sims_all = selector.text_vecs @ qv
        cand = np.argsort(-sims_all)[: selector.k_max]
        sims = sims_all[cand]
        band = adaptive_band(sims)
        cand_b, sims_b = cand[:band], sims[:band]
        if cand_b.size == 0:
            continue
        axis = axes[qi] if axes is not None else None
        X.append(features(sims_b, cand_b, selector.entities, axis,
                          selector.lat_mu, selector.lat_sd,
                          selector.en_mu, selector.en_sd))
        y.append((cand_b == targets[qi]).astype(int))
    if not X:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)
    return np.vstack(X), np.concatenate(y)
