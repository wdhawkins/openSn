#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small 2D 2G k-eigenvalue test using Power Iteration with UDSA.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        PowerIterationKEigenSolver,
        UDSAKEigenAcceleration,
    )


def make_problem():
    case_globals = globals().copy()
    case_globals["__file__"] = __file__
    with open("../transport_keigen/utils/qblock_materials.py") as f:
        exec(f.read(), case_globals)

    nodes = []
    num_cells = 10
    length = 14.0
    for i in range(num_cells + 1):
        nodes.append(length * i / num_cells)

    meshgen = OrthogonalMeshGenerator(node_sets=[nodes, nodes])
    grid = meshgen.Execute()
    grid.SetUniformBlockID(0)

    fuel = RPPLogicalVolume(
        xmin=-1000.0,
        xmax=10.0,
        ymin=-1000.0,
        ymax=10.0,
        infz=True,
    )
    grid.SetBlockIDFromLogicalVolume(fuel, 1, True)

    pquad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=8, scattering_order=0)

    return DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=case_globals["num_groups"],
        groupsets=[
            {
                "groups_from_to": (0, case_globals["num_groups"] - 1),
                "angular_quadrature": pquad,
                "inner_linear_method": "classic_richardson",
                "l_max_its": 300,
                "l_abs_tol": 1.0e-10,
            },
        ],
        xs_map=case_globals["xs_map"],
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
        ],
        options={
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": True,
            "save_angular_flux": True,
        },
    )


def run_case(use_udsa):
    phys = make_problem()
    acceleration = None
    if use_udsa:
        acceleration = UDSAKEigenAcceleration(
            problem=phys,
            verbose=False,
            max_iters=1000,
            petsc_options="-DiscreteOrdinatesProblem_UDSAksp_type preonly "
            "-DiscreteOrdinatesProblem_UDSApc_type lu",
            pi_max_its=20,
        )

    solver_args = {
        "problem": phys,
        "k_tol": 1.0e-8,
        "max_iters": 50,
    }
    if acceleration is not None:
        solver_args["acceleration"] = acceleration

    k_solver = PowerIterationKEigenSolver(**solver_args)
    k_solver.Initialize()
    k_solver.Execute()
    return k_solver.GetEigenvalue(), k_solver.GetNumIterations()


if __name__ == "__main__":
    if size != 1:
        sys.exit(f"Incorrect number of processors. Expected 1 processor but got {size}.")

    unaccelerated_k, unaccelerated_iterations = run_case(False)
    udsa_k, udsa_iterations = run_case(True)

    if rank == 0:
        print(f"small_qblock_unaccelerated_k: {unaccelerated_k:.7f}")
        print(f"small_qblock_unaccelerated_iterations: {unaccelerated_iterations}")
        print(f"small_qblock_udsa_k: {udsa_k:.7f}")
        print(f"small_qblock_udsa_iterations: {udsa_iterations}")
        print(
            "small_qblock_udsa_iteration_reduction: "
            f"{unaccelerated_iterations - udsa_iterations}"
        )
