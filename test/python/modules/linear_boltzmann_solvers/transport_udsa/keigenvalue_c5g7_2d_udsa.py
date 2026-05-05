#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual 2D C5G7 k-eigenvalue run with UDSA.

This is intentionally not listed in tests.json. It is a longer production-readiness
case intended for manual runs.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import FromFileMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        PowerIterationKEigenSolver,
        UDSAKEigenAcceleration,
    )


def make_problem(mesh_type):
    base = "../transport_keigen/c5g7"
    if mesh_type == "coarse":
        mesh_file = f"{base}/mesh/2d_c5g7_coarse.msh"
    elif mesh_type == "fine":
        mesh_file = f"{base}/mesh/2d_c5g7_refined.msh"
    else:
        raise ValueError("mesh_type must be 'coarse' or 'fine'.")

    grid = FromFileMeshGenerator(filename=mesh_file).Execute()
    grid.SetOrthogonalBoundaries()

    xs_files = [
        "XS_water.xs",
        "XS_UO2.xs",
        "XS_7pMOX.xs",
        "XS_guide_tube.xs",
        "XS_4_3pMOX.xs",
        "XS_8_7pMOX.xs",
        "XS_fission_chamber.xs",
    ]
    xss = []
    for filename in xs_files:
        xs = MultiGroupXS()
        xs.LoadFromOpenSn(f"{base}/materials/{filename}")
        xss.append(xs)

    xs_map = [{"block_ids": [mat_id], "xs": xs} for mat_id, xs in enumerate(xss)]
    num_groups = xss[0].num_groups
    quad = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=8, scattering_order=0)

    return DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": quad,
                "inner_linear_method": "petsc_gmres",
                "l_max_its": int(os.environ.get("OPENSN_C5G7_WGS_MAX_ITS", "5")),
                "l_abs_tol": 1.0e-10,
                "angle_aggregation_type": "polar",
                "angle_aggregation_num_subsets": 1,
            },
        ],
        xs_map=xs_map,
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
        ],
        options={
            "verbose_outer_iterations": True,
            "verbose_inner_iterations": True,
            "save_angular_flux": True,
            "power_default_kappa": 1.0,
        },
        sweep_type="AAH",
    )


if __name__ == "__main__":
    mesh_type = os.environ.get("OPENSN_C5G7_MESH_TYPE", "coarse")
    problem = make_problem(mesh_type)
    acceleration = UDSAKEigenAcceleration(
        problem=problem,
        verbose=bool(int(os.environ.get("OPENSN_C5G7_UDSA_VERBOSE", "0"))),
        l_abs_tol=float(os.environ.get("OPENSN_C5G7_UDSA_L_ABS_TOL", "1.0e-10")),
        max_iters=int(os.environ.get("OPENSN_C5G7_UDSA_MAX_ITERS", "200")),
        petsc_options=os.environ.get("OPENSN_C5G7_UDSA_PETSC_OPTIONS", ""),
        pi_max_its=int(os.environ.get("OPENSN_C5G7_UDSA_PI_MAX_ITS", "30")),
        pi_k_tol=1.0e-8,
        use_transport_eigenvalue=bool(
            int(os.environ.get("OPENSN_C5G7_UDSA_TRANSPORT_K", "1"))
        ),
    )
    solver = PowerIterationKEigenSolver(
        problem=problem,
        acceleration=acceleration,
        k_tol=1.0e-8,
        max_iters=int(os.environ.get("OPENSN_C5G7_PI_MAX_ITS", "100")),
    )
    solver.Initialize()
    solver.Execute()

    if rank == 0:
        print(f"c5g7_udsa_mesh_type: {mesh_type}")
        print(f"c5g7_udsa_k: {solver.GetEigenvalue():.7f}")
        print(f"c5g7_udsa_iterations: {solver.GetNumIterations()}")
