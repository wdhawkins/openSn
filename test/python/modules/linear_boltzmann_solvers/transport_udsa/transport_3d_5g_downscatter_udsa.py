#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D fixed-source 5-group downscatter test with mixed vacuum/reflecting boundaries and UDSA.
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
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume


def make_problem(apply_udsa):
    nodes = [0.0, 0.5, 1.0, 1.5]
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes, nodes, nodes])
    grid = meshgen.Execute()
    grid.SetUniformBlockID(0)
    grid.SetOrthogonalBoundaries()

    xs = MultiGroupXS()
    xs.LoadFromOpenSn("simple_5g_downscatter.xs")

    source = VolumetricSource(block_ids=[0], group_strength=[1.0, 0.0, 0.0, 0.0, 0.0])
    quad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=5,
        groupsets=[
            {
                "groups_from_to": [0, 4],
                "angular_quadrature": quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-8,
                "l_max_its": 200,
                "gmres_restart_interval": 30,
            }
        ],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[source],
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
            {"name": "zmin", "type": "reflecting"},
        ],
        options={
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": False,
            "apply_udsa": apply_udsa,
            "udsa_l_max_its": 100,
            "udsa_l_abs_tol": 1.0e-10,
            "max_ags_iterations": 80,
            "ags_tolerance": 1.0e-8,
        },
    )
    return problem


def max_scalar_fluxes(problem):
    logical_volume = RPPLogicalVolume(infx=True, infy=True, infz=True)
    values = []
    for ff in problem.GetScalarFluxFieldFunction():
        ffi = FieldFunctionInterpolationVolume()
        ffi.SetOperationType("max")
        ffi.SetLogicalVolume(logical_volume)
        ffi.AddFieldFunction(ff)
        ffi.Execute()
        values.append(ffi.GetValue())
    return values


def run_case(apply_udsa):
    problem = make_problem(apply_udsa)
    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()
    ags_summary = problem.GetAGSSolveSummary()
    if ags_summary["num_iterations"] > 0:
        num_iterations = ags_summary["num_iterations"]
    else:
        num_iterations = problem.GetWGSSolveSummary(0)["num_iterations"]
    return num_iterations, max_scalar_fluxes(problem)


if __name__ == "__main__":
    if size != 1:
        sys.exit(f"Incorrect number of processors. Expected 1 processor but got {size}.")

    unaccelerated_iterations, unaccelerated_max = run_case(False)
    udsa_iterations, udsa_max = run_case(True)

    if rank == 0:
        print(f"Downscatter-Unaccelerated-iterations={unaccelerated_iterations}")
        print(f"Downscatter-UDSA-iterations={udsa_iterations}")
        print(
            "Downscatter-UDSA-iteration-reduction="
            f"{unaccelerated_iterations - udsa_iterations}"
        )
        for g, value in enumerate(unaccelerated_max, start=1):
            print(f"Downscatter-Unaccelerated-Max-value{g}={value:.6e}")
        for g, value in enumerate(udsa_max, start=1):
            print(f"Downscatter-UDSA-Max-value{g}={value:.6e}")
