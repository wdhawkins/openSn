#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D LinearBSolver test of a block of graphite with an air cavity. DSA and TG
SDM: PWLD
Test: WGS groups [0-62] Iteration    54 Residual 7.92062e-07 CONVERGED
and   WGS groups [63-167] Iteration    63 Residual 8.1975e-07 CONVERGED
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
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSolver
    from pyopensn.logvol import RPPLogicalVolume

if __name__ == "__main__":

    num_procs = 4

    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} processors but got {size}.")

    # Setup mesh
    nodes = []
    N = 20
    L = 100.0
    xmin = -L / 2
    dx = L / N
    for i in range(N + 1):
        nodes.append(xmin + i * dx)
    znodes = [0.0, 10.0, 20.0, 30.0, 40.0]
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes, nodes, znodes])
    grid = meshgen.Execute()

    # Set block IDs
    grid.SetUniformBlockID(0)

    vol1 = RPPLogicalVolume(
        xmin=-10.0,
        xmax=10.0,
        ymin=-10.0,
        ymax=10.0,
        infz=True,
    )
    grid.SetBlockIDFromLogicalVolume(vol1, 1, True)

    # Cross sections
    num_groups = 168
    xs_graphite = MultiGroupXS()
    xs_graphite.LoadFromOpenSn("xs_graphite_pure.xs")
    xs_air = MultiGroupXS()
    xs_air.LoadFromOpenSn("xs_air50RH.xs")

    # Source
    strength = [0.0 for _ in range(num_groups)]
    strength[0] = 1.0
    mg_src0 = VolumetricSource(block_ids=[0], group_strength=strength)
    strength[0] = 0.0
    mg_src1 = VolumetricSource(block_ids=[1], group_strength=strength)

    # Setup Physics
    pquad = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=8, scattering_order=1)

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": [0, 62],
                "angular_quadrature": pquad,
                "angle_aggregation_num_subsets": 1,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-6,
                "l_max_its": 1000,
                "gmres_restart_interval": 30,
                "apply_wgdsa": True,
                "wgdsa_l_abs_tol": 1.0e-2,
            },
            {
                "groups_from_to": [63, num_groups - 1],
                "angular_quadrature": pquad,
                "angle_aggregation_num_subsets": 1,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-6,
                "l_max_its": 1000,
                "gmres_restart_interval": 30,
                "apply_wgdsa": True,
                "apply_tgdsa": True,
                "wgdsa_l_abs_tol": 1.0e-2,
            },
        ],
        xs_map=[
            {"block_ids": [0], "xs": xs_graphite},
            {"block_ids": [1], "xs": xs_air},
        ],
        scattering_order=1,
        options={
            "max_ags_iterations": 1,
            "volumetric_sources": [mg_src0, mg_src1],
        },
    )
    ss_solver = SteadyStateSolver(lbs_problem=phys)
    ss_solver.Initialize()
    ss_solver.Execute()
