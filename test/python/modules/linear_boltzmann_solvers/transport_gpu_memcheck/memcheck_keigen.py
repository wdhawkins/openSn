#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
# SPDX-License-Identifier: MIT

"""
Tiny k-eigenvalue GPU transport input for compute-sanitizer memcheck.

compute-sanitizer memcheck adds a large fixed cost to every kernel launch, so
this input runs only two power iterations, each with a single Richardson sweep
for the within-group solve. It uses a small unstructured pyramid mesh with
reflecting boundaries on all six faces (opposing reflecting pairs in every
direction), two groups, P1 scattering, and fission.

Options select the paths of the GPU source operator that the default case does
not reach: ``adjoint=True`` (transposed transfer and production matrices),
``num_groupsets=2`` (across-groupset sources and per-groupset device uploads),
and ``mixed_materials=True`` (non-fissile water in the upper half, z > 1).
With two groupsets the across-groupset solve is capped at one pass.

Neither the inner solves nor the eigenvalue are converged. The same capped
iteration sequence is run on the CPU and on the GPU and the eigenvalue and
scalar-flux vectors are compared. With identical algebra the two agree to
round-off. This is a CPU/GPU equivalence check of the GPU sweep, not an
accuracy check; accuracy of the converged eigenvalue is covered by the other
k-eigenvalue regression tests.
"""

import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    comm = MPI.COMM_WORLD

    _mpi_ops = {"sum": MPI.SUM, "max": MPI.MAX}

    def MPIAllReduce(value, op="sum"):
        return comm.allreduce(value, op=_mpi_ops[op])

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.mesh import FromFileMeshGenerator
    from pyopensn.solver import DiscreteOrdinatesProblem, PowerIterationKEigenSolver
    from pyopensn.xs import MultiGroupXS


def get_option(name, default):
    return globals().get(name, default)


def solve(sweep_type, use_gpus, adjoint, num_groupsets, mixed_materials):
    grid = FromFileMeshGenerator(filename="../../../../assets/mesh/fuel_pyramid.e").Execute()
    grid.SetOrthogonalBoundaries()

    xs = MultiGroupXS()
    xs.LoadFromOpenSn("../../../../assets/xs/xs_fuel_g2.xs")
    xs_map = [{"block_ids": [0], "xs": xs}]
    if mixed_materials:
        grid.SetBlockIDFromFunction(lambda pt, cur_id: 1 if pt.z > 1.0 else cur_id)
        water = MultiGroupXS()
        water.LoadFromOpenSn("../../../../assets/xs/xs_water_g2.xs")
        xs_map.append({"block_ids": [1], "xs": water})

    group_ranges = [(0, 1)] if num_groupsets == 1 else [(0, 0), (1, 1)]
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=2,
        groupsets=[
            {
                "groups_from_to": group_range,
                "angular_quadrature": GLCProductQuadrature3DXYZ(
                    n_polar=2,
                    n_azimuthal=4,
                    scattering_order=1,
                ),
                "angle_aggregation_type": "single",
                "inner_linear_method": "classic_richardson",
                # One sweep per inner solve; the inners are intentionally not converged.
                "l_abs_tol": 1.0e-14,
                "l_max_its": 1,
            }
            for group_range in group_ranges
        ],
        xs_map=xs_map,
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "xmax", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
            {"name": "ymax", "type": "reflecting"},
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
        ],
        # One across-groupset pass per power iteration when there are two groupsets.
        options={"max_ags_iterations": 1},
        sweep_type=sweep_type,
        use_gpus=use_gpus,
    )
    if adjoint:
        problem.SetAdjoint(True)
    # Unreachable k_tol on purpose: the solve always stops at max_iters.
    k_solver = PowerIterationKEigenSolver(problem=problem, max_iters=2, k_tol=1.0e-14)
    k_solver.Initialize()
    k_solver.Execute()
    return k_solver.GetEigenvalue(), list(problem.GetPhiNewLocal())


def compare(reference, candidate):
    """Return the global relative-L2 difference of two local flux vectors."""
    if len(reference) != len(candidate):
        raise RuntimeError("CPU and GPU scalar-flux vectors have different sizes.")

    local_diff_sq = sum((a - b) ** 2 for a, b in zip(reference, candidate))
    local_ref_sq = sum(value * value for value in reference)

    diff_norm = math.sqrt(MPIAllReduce(local_diff_sq, "sum"))
    ref_norm = math.sqrt(MPIAllReduce(local_ref_sq, "sum"))
    if not math.isfinite(ref_norm) or ref_norm <= 0.0:
        raise RuntimeError(f"CPU reference scalar flux has invalid norm {ref_norm}.")
    return diff_norm / ref_norm


if __name__ == "__main__":
    if size != 2:
        sys.exit(f"Incorrect number of processors. Expected 2 processors but got {size}.")

    sweep_type = get_option("sweep_type", "AAH")
    adjoint = get_option("adjoint", False)
    num_groupsets = get_option("num_groupsets", 1)
    mixed_materials = get_option("mixed_materials", False)
    if num_groupsets not in (1, 2):
        sys.exit(f"Unsupported num_groupsets {num_groupsets}. Expected 1 or 2.")

    options = (adjoint, num_groupsets, mixed_materials)
    cpu_k, cpu_phi = solve(sweep_type, False, *options)
    gpu_k, gpu_phi = solve(sweep_type, True, *options)
    if not math.isfinite(cpu_k) or cpu_k <= 0.0:
        raise RuntimeError(f"CPU reference eigenvalue is invalid: {cpu_k}.")
    relative_l2 = compare(cpu_phi, gpu_phi)

    if rank == 0:
        print(f"GPU_CPU_K_ABS_DIFF= {abs(cpu_k - gpu_k):.16e}")
        print(f"GPU_CPU_PHI_REL_L2_DIFF= {relative_l2:.16e}")
