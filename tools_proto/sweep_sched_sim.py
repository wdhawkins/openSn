#!/usr/bin/env python3
"""Simulate AAH sweep scheduling policies on the rank graphs of OPENSN_SWEEP_GRAPH_DUMP files.

Each rank has one device shared equally by the angle sets (one per direction) it is running. An
angle set on rank r needs `cells_r * cell_dir_cost` seconds of whole-device work and at least
`levels * tau` seconds of elapsed time. It becomes ready `lam` after all non-delayed upstream
ranks have finished the same direction (the AAH gate). Policies choose which ready angle sets run:

  all       every ready angle set starts at once (the fused scheduler with wide runs)
  fifo:K    at most K in flight, in direction order
  prio:K    at most K in flight, longest remaining downstream rank chain first

Usage: sweep_sched_sim.py <prefix> --cells N [--policies all fifo:32 prio:32] [--dt 1e-4]
"""

import argparse
import sys
from collections import defaultdict

from sweep_graph_model import read_rank, topo_order
import glob


def simulate(data, ranks, num_dirs, cells_per_rank, cost, tau, lam, policy, dt):
    kind, _, k = policy.partition(":")
    limit = int(k) if k else None
    deps = {(r, d): data[r][d]["deps"] for r in ranks for d in range(num_dirs)}
    succ = defaultdict(list)
    for (r, d), ps in deps.items():
        for p in ps:
            succ[(p, d)].append(r)
    # Remaining downstream chain length (in ranks) of each (rank, direction).
    chain = {}
    for d in range(num_dirs):
        order = topo_order(ranks, {r: deps[(r, d)] for r in ranks})
        for r in reversed(order):
            chain[(r, d)] = 1 + max((chain[(s, d)] for s in succ[(r, d)]), default=0)

    work = {r: cells_per_rank[r] * cost for r in ranks}
    waiting = {(r, d): len(deps[(r, d)]) for r in ranks for d in range(num_dirs)}
    ready_at = {key: 0.0 for key, n in waiting.items() if n == 0}
    running = {r: {} for r in ranks}  # (d) -> [work_done, start_time]
    done = set()
    t = 0.0
    total = len(waiting)
    while len(done) < total:
        # Admit ready angle sets.
        for r in ranks:
            cand = [d for (rr, d), ta in ready_at.items() if rr == r and ta <= t and d not in running[r]
                    and (r, d) not in done]
            if not cand:
                continue
            if kind == "prio":
                cand.sort(key=lambda d: (-chain[(r, d)], d))
            else:
                cand.sort()
            room = len(cand) if limit is None else max(0, limit - len(running[r]))
            for d in cand[:room]:
                running[r][d] = [0.0, t]
                del ready_at[(r, d)]
        # Advance by dt with equal device sharing.
        t += dt
        for r in ranks:
            n = len(running[r])
            if n == 0:
                continue
            finished = []
            for d, state in running[r].items():
                state[0] += dt / n
                if state[0] >= work[r] and t - state[1] >= data[r][d]["levels"] * tau:
                    finished.append(d)
            for d in finished:
                del running[r][d]
                done.add((r, d))
                for s in succ[(r, d)]:
                    waiting[(s, d)] -= 1
                    if waiting[(s, d)] == 0:
                        ready_at[(s, d)] = t + lam
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--cells", type=int, required=True, help="global cell count")
    ap.add_argument("--policies", nargs="+", default=["all", "fifo:32", "prio:32"])
    ap.add_argument("--tau", type=float, default=25e-6)
    ap.add_argument("--lam", type=float, default=50e-6)
    ap.add_argument("--cell-dir-cost", type=float, default=20.4e-9)
    ap.add_argument("--dt", type=float, default=1e-3)
    args = ap.parse_args()
    data = dict(read_rank(f) for f in sorted(glob.glob(f"{args.prefix}.r*.bin")))
    ranks = sorted(data)
    num_dirs = len(data[ranks[0]])
    cells = {r: args.cells / len(ranks) for r in ranks}
    bound = args.cells / len(ranks) * num_dirs * args.cell_dir_cost
    print(f"{args.prefix}: ranks {len(ranks)}, work bound {bound:.3f} s")
    for policy in args.policies:
        t = simulate(data, ranks, num_dirs, cells, args.cell_dir_cost, args.tau, args.lam, policy,
                     args.dt)
        print(f"  {policy:>10}: {t:.3f} s ({t / bound:.2f}x work bound)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
