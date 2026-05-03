#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Transport test with one groupset and UDSA
SDM: PWLD
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import FromFileMeshGenerator, ExtruderMeshGenerator, KBAGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume


def make_grid():
    meshgen = ExtruderMeshGenerator(
        inputs=[
            FromFileMeshGenerator(
                filename="../../../../assets/mesh/triangle_mesh2x2_cuts.obj"
            )
        ],
        layers=[
            {"z": 0.4, "n": 2},
            {"z": 0.8, "n": 2},
            {"z": 1.2, "n": 2},
            {"z": 1.6, "n": 2},
        ],
        partitioner=KBAGraphPartitioner(nx=2, ny=2, xcuts=[0.0], ycuts=[0.0]),
    )
    return meshgen.Execute()


def make_groupsets(pquad):
    return [
        {
            "groups_from_to": [0, 2],
            "angular_quadrature": pquad,
            "inner_linear_method": "petsc_gmres",
            "l_abs_tol": 1.0e-6,
            "l_max_its": 300,
            "gmres_restart_interval": 30,
        }
    ]


def max_scalar_fluxes(problem, logical_volume):
    values = []
    for ff in problem.GetScalarFluxFieldFunction():
        ffi = FieldFunctionInterpolationVolume()
        ffi.SetOperationType("max")
        ffi.SetLogicalVolume(logical_volume)
        ffi.AddFieldFunction(ff)
        ffi.Execute()
        values.append(ffi.GetValue())
    return values


def effective_iteration_count(problem):
    ags_summary = problem.GetAGSSolveSummary()
    if ags_summary["num_iterations"] > 0:
        return ags_summary["num_iterations"]
    return problem.GetWGSSolveSummary(0)["num_iterations"]


def run_case(grid, logical_volume, xs, source, quadrature, apply_udsa):
    options = {
        "verbose_inner_iterations": False,
        "verbose_outer_iterations": False,
        "apply_udsa": apply_udsa,
        "udsa_l_max_its": 20,
        "udsa_l_abs_tol": 1.0e-8,
        "max_ags_iterations": 30,
        "ags_tolerance": 1.0e-6,
    }
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=3,
        groupsets=make_groupsets(quadrature),
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[source],
        options=options,
    )
    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()
    return effective_iteration_count(problem), max_scalar_fluxes(problem, logical_volume)


if __name__ == "__main__":
    num_procs = 4
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} processors but got {size}.")

    grid = make_grid()
    vol0 = RPPLogicalVolume(infx=True, infy=True, infz=True)
    grid.SetBlockIDFromLogicalVolume(vol0, 0, True)

    xs_upscatter = MultiGroupXS()
    xs_upscatter.LoadFromOpenSn("simple_upscatter.xs")

    mg_src = VolumetricSource(block_ids=[0], group_strength=[1.0, 1.0, 1.0])
    pquad = GLCProductQuadrature3DXYZ(n_polar=4, n_azimuthal=8, scattering_order=0)

    unaccelerated_iterations, unaccelerated_max = run_case(
        grid, vol0, xs_upscatter, mg_src, pquad, False
    )
    udsa_iterations, udsa_max = run_case(grid, vol0, xs_upscatter, mg_src, pquad, True)

    if rank == 0:
        print(f"Unaccelerated-iterations={unaccelerated_iterations}")
        print(f"UDSA-iterations={udsa_iterations}")
        print(f"UDSA-iteration-reduction={unaccelerated_iterations - udsa_iterations}")
        for g, value in enumerate(unaccelerated_max, start=1):
            print(f"Unaccelerated-Max-value{g}={value:.5e}")
        for g, value in enumerate(udsa_max, start=1):
            print(f"UDSA-Max-value{g}={value:.5e}")
