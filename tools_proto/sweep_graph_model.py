#!/usr/bin/env python3
"""Critical-path model of AAH sweeps from OPENSN_SWEEP_GRAPH_DUMP files.

For each direction (SPDS), the ranks form a DAG through their non-delayed predecessor locations.
Two start rules are compared with infinite device capacity:

  gate     a rank starts a direction after all upstream ranks have finished it (current AAH),
           then sweeps its D levels at tau per level.
  chunked  a rank's level L finishes tau after both its level L-1 and the arrival of every
           upstream face that level L reads. A face arrives lam after the upstream level that
           wrote it finishes.

The sweep time cannot be below the larger of the slowest direction's finish (critical path) and
the busiest rank's work at single-device throughput.

Usage: sweep_graph_model.py <prefix> [--tau 25e-6] [--lam 50e-6] [--cell-dir-cost 20.4e-9]
"""

import argparse
import glob
import struct
import sys
from collections import defaultdict


def read_rank(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from(fmt, data, off)
        off += struct.calcsize(fmt)
        return vals

    rank, num_spds = take("<ii")
    spds = []
    for _ in range(num_spds):
        (num_levels,) = take("<i")
        (nd,) = take("<i")
        deps = list(take(f"<{nd}i")) if nd else []
        (ndl,) = take("<i")
        delayed = list(take(f"<{ndl}i")) if ndl else []
        (nout,) = take("<i")
        out = {}
        for _ in range(nout):
            gid, level = take("<Qi")
            out[gid] = level
        (nin,) = take("<i")
        incoming = [take("<Qii") for _ in range(nin)]
        spds.append(dict(levels=num_levels, deps=deps, delayed=set(delayed), out=out, inc=incoming))
    return rank, spds


def topo_order(ranks, deps):
    indeg = {r: len(deps[r]) for r in ranks}
    succ = defaultdict(list)
    for r in ranks:
        for p in deps[r]:
            succ[p].append(r)
    order = [r for r in ranks if indeg[r] == 0]
    i = 0
    while i < len(order):
        for s in succ[order[i]]:
            indeg[s] -= 1
            if indeg[s] == 0:
                order.append(s)
        i += 1
    if len(order) != len(ranks):
        raise RuntimeError("non-delayed rank graph has a cycle")
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--tau", type=float, default=25e-6)
    ap.add_argument("--lam", type=float, default=50e-6)
    ap.add_argument("--cell-dir-cost", type=float, default=20.4e-9,
                    help="seconds per cell-direction (all groups) at single-device throughput")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.prefix}.r*.bin"))
    data = dict(read_rank(f) for f in files)
    ranks = sorted(data)
    num_dirs = len(data[ranks[0]])
    tau, lam = args.tau, args.lam

    gate_finish = []
    chunk_finish = []
    gate_ranks_on_path = []
    for d in range(num_dirs):
        sp = {r: data[r][d] for r in ranks}
        deps = {r: [p for p in sp[r]["deps"]] for r in ranks}
        order = topo_order(ranks, deps)

        # Gate model.
        fin = {}
        for r in order:
            start = max((fin[p] + lam for p in deps[r]), default=0.0)
            fin[r] = start + sp[r]["levels"] * tau
        gate_finish.append(max(fin.values()))

        # Chunked model: per-rank level finish times.
        level_end = {}
        for r in order:
            D = sp[r]["levels"]
            arrive = [0.0] * D
            for up_gid, up_rank, my_level in sp[r]["inc"]:
                if up_rank in sp[r]["delayed"] or up_rank not in level_end:
                    continue
                up_level = data[up_rank][d]["out"].get(up_gid)
                if up_level is None or my_level < 0:
                    continue
                t = level_end[up_rank][up_level] + lam
                if t > arrive[my_level]:
                    arrive[my_level] = t
            ends = []
            prev = 0.0
            for L in range(D):
                prev = max(prev, arrive[L]) + tau
                ends.append(prev)
            level_end[r] = ends
        chunk_finish.append(max((e[-1] for e in level_end.values() if e), default=0.0))
        depth = {}
        for r in order:
            depth[r] = 1 + max((depth[p] for p in deps[r]), default=0)
        gate_ranks_on_path.append(max(depth.values()))

    # Work bound: cells per rank are not in the dump, so use levels' total via outgoing counts is
    # not enough. Approximate cells per rank from the mesh size given on the command line instead.
    P = len(ranks)
    max_levels = max(data[r][d]["levels"] for r in ranks for d in range(num_dirs))
    avg_levels = sum(data[r][d]["levels"] for r in ranks for d in range(num_dirs)) / (P * num_dirs)
    print(f"{args.prefix}: ranks {P}, directions {num_dirs}, levels per rank-direction "
          f"avg {avg_levels:.0f} max {max_levels}, rank-chain length avg "
          f"{sum(gate_ranks_on_path) / num_dirs:.2f} max {max(gate_ranks_on_path)}")
    print(f"  critical path (tau {tau * 1e6:.0f} us, lam {lam * 1e6:.0f} us): "
          f"gate {max(gate_finish) * 1e3:.2f} ms, chunked {max(chunk_finish) * 1e3:.2f} ms, "
          f"ratio {max(gate_finish) / max(chunk_finish):.2f}")
    print(f"  mean over directions: gate {sum(gate_finish) / num_dirs * 1e3:.2f} ms, chunked "
          f"{sum(chunk_finish) / num_dirs * 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
