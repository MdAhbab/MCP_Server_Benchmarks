"""
Operational-metadata collection and attestation for MCP tool definitions.

The discovery experiments show that latency and energy metadata, not index
structure, is what lets an agent choose the least costly of several equivalent
tools. No public MCP server publishes those fields, so the gain is currently
unreachable. This module is the missing half of that recommendation: a way to
measure the fields and a way to publish them so a consumer can decide whether to
believe them.

Self-declared performance numbers invite inflation, so a bare field in a tool
definition is not enough. The record here binds three things together:

  what was measured   per-call latency percentiles and energy per call
  how it was measured harness, sample count, concurrency, hardware, timestamp
  who measured it     an Ed25519 signature over the canonical record

A consumer verifies the signature, then decides whether it trusts the signer and
whether the stated conditions resemble its own. A server operator signing its
own numbers is still making a checkable claim: the record is non-repudiable, and
an independent measurer can publish a competing record over the same tool.

    python attestation.py --demo
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

SCHEMA = "mcp-operational-attestation/v1"


@dataclass
class MeasurementConditions:
    """Everything a consumer needs to judge whether a record applies to it."""
    harness: str
    samples: int
    concurrency: int
    hardware: str
    platform: str
    timestamp_utc: str
    energy_instrument: str = "none"
    notes: str = ""


@dataclass
class OperationalRecord:
    server: str
    tool: str
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    energy_j_per_call: float | None
    conditions: MeasurementConditions
    schema: str = SCHEMA

    def canonical(self) -> bytes:
        """Byte form that gets signed. Sorted keys, no incidental whitespace."""
        d = asdict(self)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()


@dataclass
class Attestation:
    record: dict
    digest_sha256: str
    signature_b64: str
    public_key_b64: str
    algorithm: str = "ed25519"
    schema: str = SCHEMA


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def measure_tool(call: Callable[[], object], samples: int = 30,
                 warmup: int = 3, energy_meter=None) -> dict:
    """Time repeated calls to one tool and, if available, meter their energy.

    Warm-up calls are discarded so first-call effects such as connection setup
    and lazy imports do not enter the distribution.
    """
    for _ in range(warmup):
        try:
            call()
        except Exception:
            pass

    lat, errors = [], 0
    e0 = energy_meter.read_j() if energy_meter is not None else None
    t0 = time.perf_counter()
    for _ in range(samples):
        s = time.perf_counter()
        try:
            call()
        except Exception:
            errors += 1
        lat.append((time.perf_counter() - s) * 1000.0)
    elapsed = time.perf_counter() - t0
    e1 = energy_meter.read_j() if energy_meter is not None else None

    lat.sort()
    def pct(p):
        return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))]
    energy = ((e1 - e0) / samples) if (e0 is not None and e1 is not None) else None
    return {
        "p50": pct(50), "p95": pct(95), "p99": pct(99),
        "mean": statistics.mean(lat), "errors": errors,
        "energy_j_per_call": energy, "wall_seconds": elapsed,
    }


def build_record(server: str, tool: str, stats: dict, harness: str,
                 samples: int, concurrency: int = 1,
                 energy_instrument: str = "none") -> OperationalRecord:
    return OperationalRecord(
        server=server, tool=tool,
        latency_ms_p50=round(stats["p50"], 3),
        latency_ms_p95=round(stats["p95"], 3),
        latency_ms_p99=round(stats["p99"], 3),
        energy_j_per_call=(round(stats["energy_j_per_call"], 6)
                           if stats.get("energy_j_per_call") is not None else None),
        conditions=MeasurementConditions(
            harness=harness, samples=samples, concurrency=concurrency,
            hardware=platform.processor() or platform.machine(),
            platform=platform.platform(),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            energy_instrument=energy_instrument,
        ),
    )


# ---------------------------------------------------------------------------
# signing and verification
# ---------------------------------------------------------------------------

def generate_keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    sk = Ed25519PrivateKey.generate()
    return sk, sk.public_key()


def sign(record: OperationalRecord, private_key) -> Attestation:
    from cryptography.hazmat.primitives import serialization
    payload = record.canonical()
    sig = private_key.sign(payload)
    pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    return Attestation(
        record=json.loads(payload),
        digest_sha256=record.digest(),
        signature_b64=base64.b64encode(sig).decode(),
        public_key_b64=base64.b64encode(pub).decode(),
    )


def verify(att: Attestation | dict) -> tuple[bool, str]:
    """Check the signature and that the digest matches the record as published.

    Returns (ok, reason). A false result distinguishes a tampered record from a
    bad signature, because the two failures mean different things to a consumer.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    d = asdict(att) if isinstance(att, Attestation) else dict(att)
    payload = json.dumps(d["record"], sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(payload).hexdigest() != d["digest_sha256"]:
        return False, "record does not match its digest: it was altered after signing"
    try:
        pk = Ed25519PublicKey.from_public_bytes(base64.b64decode(d["public_key_b64"]))
        pk.verify(base64.b64decode(d["signature_b64"]), payload)
    except Exception as e:
        return False, f"signature check failed: {type(e).__name__}"
    return True, "signature valid for this record and key"


# ---------------------------------------------------------------------------
# demonstration
# ---------------------------------------------------------------------------

def demo() -> dict:
    """Measure two stand-in tools, sign the records, then try to forge one."""
    import random
    rng = random.Random(7)

    def fast_tool():
        time.sleep(0.004 + rng.random() * 0.002)

    def slow_tool():
        time.sleep(0.020 + rng.random() * 0.008)

    out = {"schema": SCHEMA, "records": [], "verification": []}
    sk, _pk = generate_keypair()

    for name, fn in (("fast.search", fast_tool), ("slow.search", slow_tool)):
        stats = measure_tool(fn, samples=25)
        rec = build_record("demo-server", name, stats,
                           harness="attestation.py demo", samples=25)
        att = sign(rec, sk)
        ok, why = verify(att)
        out["records"].append(asdict(att))
        out["verification"].append({"tool": name, "valid": ok, "reason": why})

    # A consumer that trusts the field blindly can be lied to. Alter a signed
    # record's latency downward and confirm verification rejects it.
    forged = json.loads(json.dumps(out["records"][1]))
    forged["record"]["latency_ms_p50"] = 0.1
    ok, why = verify(forged)
    out["forgery_check"] = {
        "attempt": "rewrote latency_ms_p50 on a signed record",
        "accepted": ok,
        "reason": why,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results/attestation_demo.json")
    a = ap.parse_args()
    if a.demo:
        res = demo()
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        for v in res["verification"]:
            print(f"  {v['tool']:14s} valid={v['valid']}  {v['reason']}")
        f = res["forgery_check"]
        print(f"  forged record accepted={f['accepted']}  {f['reason']}")
        print(f"wrote {a.out}")
