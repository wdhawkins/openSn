#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Exact infinite-medium transport on a mixed triangle/quadrilateral mesh.

With reflecting boundaries, uniform isotropic sources, no scattering, and
unit total cross section, every group's exact scalar flux is its source
strength. The mixed mesh forces the AAH sweep to dispatch its cell solve per
cell instead of selecting one fixed-node-count kernel for the entire mesh.
Nine groups exercise both SIMD batches and a remainder on supported coverage
hosts without changing the analytic solution.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import FromFileMeshGenerator
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS


def field_extreme(field_function, operation, volume):
    interpolator = FieldFunctionInterpolationVolume()
    interpolator.SetOperationType(operation)
    interpolator.SetLogicalVolume(volume)
    interpolator.AddFieldFunction(field_function)
    interpolator.Execute()
    return interpolator.GetValue()


if __name__ == "__main__":
    grid = FromFileMeshGenerator(
        filename="../../../../assets/mesh/mixed_tri_quad_2d.obj"
    ).Execute()
    grid.SetOrthogonalBoundaries()
    grid.SetUniformBlockID(0)

    num_groups = 9
    source_strengths = [float(g + 1) for g in range(num_groups)]
    xs = MultiGroupXS()
    xs.LoadFromOpenSn("../../../../assets/xs/pure_absorber_9g.xs")

    quadrature = GLCProductQuadrature2DXY(
        n_polar=4, n_azimuthal=8, scattering_order=0
    )
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": quadrature,
                "angle_aggregation_type": "single",
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-12,
                "l_max_its": 100,
            }
        ],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[
            VolumetricSource(block_ids=[0], group_strength=source_strengths)
        ],
        boundary_conditions=[
            {"name": name, "type": "reflecting"}
            for name in ("xmin", "xmax", "ymin", "ymax")
        ],
        options={"verbose_inner_iterations": False},
    )

    solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
    solver.Initialize()
    solver.Execute()

    whole_domain = RPPLogicalVolume(infx=True, infy=True, infz=True)
    max_relative_error = 0.0
    for exact, field_function in zip(
        source_strengths, problem.GetScalarFluxFieldFunction()
    ):
        minimum = field_extreme(field_function, "min", whole_domain)
        maximum = field_extreme(field_function, "max", whole_domain)
        max_relative_error = max(
            max_relative_error,
            abs(minimum / exact - 1.0),
            abs(maximum / exact - 1.0),
        )

    balance = solver.ComputeBalanceTable()
    balance_error = abs(balance["production_rate"] - balance["absorption_rate"])

    if rank == 0:
        print(f"MIXED_CELL_MAX_REL_ERROR={max_relative_error:.12e}")
        print(f"MIXED_CELL_BALANCE_ABS_ERROR={balance_error:.12e}")
