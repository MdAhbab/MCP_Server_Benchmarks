"""
Builds a compact catalogue from the MCP-tools dataset released with MCP-Zero.

Source: Fei, Zheng and Feng, "MCP-Zero: Active Tool Discovery for Autonomous LLM
Agents", arXiv:2506.01056. Dataset of 308 servers and 2,797 tools drawn from the
official Model Context Protocol repository.

The published file is a ~333 MB JSON array of servers, most of which is precomputed
sentence embeddings. This script streams it, drops the server-level embeddings, and
writes two artefacts:

    data/mcp_tools_catalogue.json   name, description, parameters, server, per tool
    data/mcp_tools_embeddings.npy   float32 matrix of the released tool embeddings

Download the dataset first (the project's Git LFS quota is exhausted, so the authors
distribute it via Google Drive):

    https://github.com/xfey/MCP-Zero/blob/master/MCP-tools/download_data.md

then run:

    python mcp_zero_catalogue.py path/to/downloaded.json
"""

import json
import os
import sys

import numpy as np


def stream_servers(path):
    """Yield one server object at a time without loading the whole file."""
    with open(path, encoding="utf-8") as f:
        depth = 0
        buf = []
        in_str = False
        esc = False
        while True:
            ch = f.read(1 << 20)
            if not ch:
                break
            for c in ch:
                if depth:
                    buf.append(c)
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == "{":
                    if not depth:
                        buf = ["{"]
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads("".join(buf))
                        buf = []


def build(src, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    catalogue, embeddings = [], []
    n_servers = 0
    for srv in stream_servers(src):
        n_servers += 1
        server = srv.get("name") or "unknown"
        for t in srv.get("tools") or []:
            desc = (t.get("description") or "").strip()
            name = (t.get("name") or "").strip()
            if not name:
                continue
            params = t.get("parameter") or {}
            catalogue.append({
                "name": name,
                "description": desc,
                "parameters": params,
                "server": server,
                "server_summary": (srv.get("summary") or "")[:400],
            })
            emb = t.get("description_embedding")
            embeddings.append(emb if emb else None)

    dim = next((len(e) for e in embeddings if e), 0)
    mat = np.zeros((len(embeddings), dim), dtype=np.float32)
    have = 0
    for i, e in enumerate(embeddings):
        if e:
            mat[i] = np.asarray(e, dtype=np.float32)
            have += 1

    cpath = os.path.join(out_dir, "mcp_tools_catalogue.json")
    epath = os.path.join(out_dir, "mcp_tools_embeddings.npy")
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, ensure_ascii=False)
    np.save(epath, mat)

    print(f"servers parsed      : {n_servers}")
    print(f"tools extracted     : {len(catalogue)}")
    print(f"tools with embedding: {have} (dim {dim})")
    print(f"wrote {cpath}")
    print(f"wrote {epath}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1])
