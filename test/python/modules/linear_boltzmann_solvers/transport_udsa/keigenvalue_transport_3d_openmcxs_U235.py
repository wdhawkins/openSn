#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D 84G k-eigenvalue test using OpenMC MGXS cross sections and UDSA.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        PowerIterationKEigenSolver,
        UDSAKEigenAcceleration,
    )


def make_problem():
    nx = 5
    length = 2.0
    nodes = [length * i / nx for i in range(nx + 1)]
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes])
    grid = meshgen.Execute()
    grid.SetUniformBlockID(0)

    num_groups = 84
    xs_u235 = MultiGroupXS()
    xs_u235.LoadFromOpenMC("../transport_keigen/u235_84g.h5", "set1", 294.0)

    return DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": GLCProductQuadrature3DXYZ(
                    n_polar=2,
                    n_azimuthal=4,
                    scattering_order=0,
                ),
                "inner_linear_method": "classic_richardson",
                "l_max_its": 2,
                "l_abs_tol": 1.0e-12,
            },
        ],
        xs_map=[
            {"block_ids": [0], "xs": xs_u235},
        ],
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "xmax", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
            {"name": "ymax", "type": "reflecting"},
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
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
            verbose=bool(int(os.environ.get("OPENSN_U235_UDSA_VERBOSE", "0"))),
            max_iters=int(os.environ.get("OPENSN_U235_UDSA_MAX_ITERS", "200")),
            petsc_options=os.environ.get(
                "OPENSN_U235_UDSA_PETSC_OPTIONS",
                "-DiscreteOrdinatesProblem_UDSAksp_type gmres "
                "-DiscreteOrdinatesProblem_UDSApc_type hypre",
            ),
            pi_k_tol=1.0e-8,
            pi_max_its=int(os.environ.get("OPENSN_U235_UDSA_PI_MAX_ITS", "30")),
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
    num_procs = int(os.environ.get("OPENSN_U235_NUM_PROCS", "4"))
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} but got {size}.")

    run_unaccelerated = bool(int(os.environ.get("OPENSN_U235_RUN_UNACCELERATED", "1")))
    if run_unaccelerated:
        unaccelerated_k, unaccelerated_iterations = run_case(False)
    else:
        unaccelerated_k = float("nan")
        unaccelerated_iterations = 0

    udsa_k, udsa_iterations = run_case(True)

    if rank == 0:
        print(f"u235_84g_unaccelerated_k: {unaccelerated_k:.7f}")
        print(f"u235_84g_unaccelerated_iterations: {unaccelerated_iterations}")
        print(f"u235_84g_udsa_k: {udsa_k:.7f}")
        print(f"u235_84g_udsa_iterations: {udsa_iterations}")
        print(
            "u235_84g_udsa_iteration_reduction: "
            f"{unaccelerated_iterations - udsa_iterations}"
        )
