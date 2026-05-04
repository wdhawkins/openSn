#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D 2G KEigenvalue::Solver test using Power Iteration with UDSA
Test: Final k-eigenvalue: 0.5969127
"""

import os
import sys
import math

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem, PowerIterationKEigenSolver
    from pyopensn.solver import UDSAKEigenAcceleration

def make_problem():
    case_globals = globals().copy()
    case_globals["__file__"] = __file__
    with open("../transport_keigen/utils/qblock_mesh.py") as f:
        exec(f.read(), case_globals)
    with open("../transport_keigen/utils/qblock_materials.py") as f:
        exec(f.read(), case_globals)

    # Setup Physics
    pquad = GLCProductQuadrature2DXY(n_polar=8, n_azimuthal=16, scattering_order=2)

    return DiscreteOrdinatesProblem(
        mesh=case_globals["grid"],
        num_groups=case_globals["num_groups"],
        groupsets=[
            {
                "groups_from_to": (0, case_globals["num_groups"] - 1),
                "angular_quadrature": pquad,
                "inner_linear_method": "classic_richardson",
                "l_max_its": 500,
                "gmres_restart_interval": 50,
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
    unaccelerated_k, unaccelerated_iterations = run_case(False)
    udsa_k, udsa_iterations = run_case(True)

    if rank == 0:
        print(f"qblock_unaccelerated_k: {unaccelerated_k:.7f}")
        print(f"qblock_unaccelerated_iterations: {unaccelerated_iterations}")
        print(f"qblock_udsa_k: {udsa_k:.7f}")
        print(f"qblock_udsa_iterations: {udsa_iterations}")
        print(
            "qblock_udsa_iteration_reduction: "
            f"{unaccelerated_iterations - udsa_iterations}"
        )
