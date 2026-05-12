#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if "opensn_console" not in globals():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLProductQuadrature1DSlab
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver


def write_multigroup_xs(xs_path):
    if rank == 0:
        with open(xs_path, "w", encoding="ascii") as xs_file:
            xs_file.write(
                """NUM_GROUPS 2
NUM_MOMENTS 1

SIGMA_T_BEGIN
0 1.00
1 1.10
SIGMA_T_END

SIGMA_A_BEGIN
0 0.35
1 0.45
SIGMA_A_END

TRANSFER_MOMENTS_BEGIN
M_GFROM_GTO_VAL 0 0 0 0.50
M_GFROM_GTO_VAL 0 0 1 0.10
M_GFROM_GTO_VAL 0 1 0 0.15
M_GFROM_GTO_VAL 0 1 1 0.45
TRANSFER_MOMENTS_END
"""
            )
    comm.Barrier()


def make_mesh(num_cells=24):
    nodes = [i / num_cells for i in range(num_cells + 1)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes]).Execute()
    grid.SetUniformBlockID(0)
    return grid


def make_single_groupset_problem(restart_read="", restart_write="", l_max_its=200):
    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(1.0, 0.8)
    quad = GLProductQuadrature1DSlab(n_polar=8, scattering_order=0)
    source = VolumetricSource(block_ids=[0], group_strength=[1.0])

    options = {
        "verbose_inner_iterations": False,
        "verbose_outer_iterations": False,
        "max_ags_iterations": 1,
    }
    if restart_read:
        options["read_restart_path"] = restart_read
    if restart_write:
        options["restart_writes_enabled"] = True
        options["write_restart_path"] = restart_write

    return DiscreteOrdinatesProblem(
        mesh=make_mesh(),
        num_groups=1,
        groupsets=[
            {
                "groups_from_to": (0, 0),
                "angular_quadrature": quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-10,
                "l_max_its": l_max_its,
                "gmres_restart_interval": 20,
            }
        ],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[source],
        boundary_conditions=[
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
        ],
        options=options,
    )


def make_multi_groupset_problem(xs_path, restart_read="", restart_write="", max_ags_iterations=80):
    xs = MultiGroupXS()
    xs.LoadFromOpenSn(xs_path)
    quad = GLProductQuadrature1DSlab(n_polar=8, scattering_order=0)
    source = VolumetricSource(block_ids=[0], group_strength=[1.0, 0.25])

    options = {
        "verbose_inner_iterations": False,
        "verbose_outer_iterations": False,
        "max_ags_iterations": max_ags_iterations,
        "ags_tolerance": 1.0e-10,
    }
    if restart_read:
        options["read_restart_path"] = restart_read
    if restart_write:
        options["restart_writes_enabled"] = True
        options["write_restart_path"] = restart_write

    return DiscreteOrdinatesProblem(
        mesh=make_mesh(),
        num_groups=2,
        groupsets=[
            {
                "groups_from_to": (0, 0),
                "angular_quadrature": quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-11,
                "l_max_its": 100,
                "gmres_restart_interval": 20,
            },
            {
                "groups_from_to": (1, 1),
                "angular_quadrature": quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-11,
                "l_max_its": 100,
                "gmres_restart_interval": 20,
            },
        ],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[source],
        boundary_conditions=[
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
        ],
        options=options,
    )


def solve(problem):
    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()
    return (
        list(problem.GetPhiNewLocal()),
        int(solver.GetLastAGSSolveIterations()),
        [int(x) for x in solver.GetLastWGSSolveIterations()],
    )


def global_max_abs_diff(v1, v2):
    local_diff = 0.0
    for a, b in zip(v1, v2):
        local_diff = max(local_diff, abs(a - b))
    return comm.allreduce(local_diff, op=MPI.MAX)


def restart_files_exist(stem):
    return comm.allreduce(int(os.path.exists(f"{stem}{rank}.restart.h5")), op=MPI.MIN) == 1


if __name__ == "__main__":
    num_procs = 1
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} but got {size}.")

    restart_dir = "steady_state_restart_wgs_ags"
    single_stem = os.path.join(restart_dir, "single")
    multi_stem = os.path.join(restart_dir, "multi")
    xs_path = os.path.join(restart_dir, "two_group_steady.xs")

    if rank == 0 and os.path.isdir(restart_dir):
        shutil.rmtree(restart_dir)
    if rank == 0:
        os.makedirs(restart_dir, exist_ok=True)
    comm.Barrier()
    write_multigroup_xs(xs_path)

    single_ref, single_ref_ags, single_ref_wgs = solve(make_single_groupset_problem())
    solve(make_single_groupset_problem(restart_write=single_stem, l_max_its=3))
    single_restart, single_restart_ags, single_restart_wgs = solve(
        make_single_groupset_problem(restart_read=single_stem)
    )
    single_diff = global_max_abs_diff(single_ref, single_restart)
    single_saved_iterations = single_ref_wgs[0] - single_restart_wgs[0]
    single_pass = int(
        single_diff < 1.0e-8
        and restart_files_exist(single_stem)
        and single_saved_iterations > 0
    )

    multi_ref, multi_ref_ags, multi_ref_wgs = solve(make_multi_groupset_problem(xs_path))
    solve(make_multi_groupset_problem(xs_path, restart_write=multi_stem, max_ags_iterations=1))
    multi_restart, multi_restart_ags, multi_restart_wgs = solve(
        make_multi_groupset_problem(xs_path, restart_read=multi_stem)
    )
    multi_diff = global_max_abs_diff(multi_ref, multi_restart)
    multi_saved_iterations = multi_ref_ags - multi_restart_ags
    multi_pass = int(
        multi_diff < 1.0e-8
        and restart_files_exist(multi_stem)
        and multi_saved_iterations > 0
    )

    if rank == 0:
        print(f"STEADY_SINGLE_GROUPSET_RESTART_MAX_ABS_DIFF={single_diff:.16e}")
        print(f"STEADY_SINGLE_GROUPSET_REFERENCE_WGS_ITS={single_ref_wgs[0]}")
        print(f"STEADY_SINGLE_GROUPSET_RESTART_WGS_ITS={single_restart_wgs[0]}")
        print(f"STEADY_SINGLE_GROUPSET_RESTART_SAVED_WGS_ITS={single_saved_iterations}")
        print(f"STEADY_SINGLE_GROUPSET_RESTART_PASS={single_pass}")
        print(f"STEADY_MULTI_GROUPSET_RESTART_MAX_ABS_DIFF={multi_diff:.16e}")
        print(f"STEADY_MULTI_GROUPSET_REFERENCE_AGS_ITS={multi_ref_ags}")
        print(f"STEADY_MULTI_GROUPSET_RESTART_AGS_ITS={multi_restart_ags}")
        print(f"STEADY_MULTI_GROUPSET_RESTART_SAVED_AGS_ITS={multi_saved_iterations}")
        print(f"STEADY_MULTI_GROUPSET_RESTART_PASS={multi_pass}")

    comm.Barrier()
    if rank == 0 and os.path.isdir(restart_dir):
        shutil.rmtree(restart_dir)
