#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Verify a distributed PVTU round trip with an exact transport solution.

The original nonuniform mesh has analytically known block areas. After export
and reimport, those areas must be preserved. With reflecting boundaries, unit
total cross section, no scattering, and a unit isotropic source, the exact
scalar flux is one everywhere and production equals absorption.
"""

import os
from pathlib import Path
import sys

if "opensn_console" in globals():
    barrier = MPIBarrier
else:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    barrier = MPI.COMM_WORLD.Barrier
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import (
        FromFileMeshGenerator,
        LinearGraphPartitioner,
        OrthogonalMeshGenerator,
    )
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS


def remove_output(base_name):
    Path(f"{base_name}_{rank}.vtu").unlink(missing_ok=True)
    if rank == 0:
        Path(f"{base_name}.pvtu").unlink(missing_ok=True)
    barrier()


def field_extreme(field_function, operation, volume):
    interpolator = FieldFunctionInterpolationVolume()
    interpolator.SetOperationType(operation)
    interpolator.SetLogicalVolume(volume)
    interpolator.AddFieldFunction(field_function)
    interpolator.Execute()
    return interpolator.GetValue()


if __name__ == "__main__":
    if size != 2:
        raise RuntimeError(f"PVTU round-trip test requires two ranks, got {size}.")

    output_base = "pvtu_roundtrip_transport"
    remove_output(output_base)

    grid = OrthogonalMeshGenerator(
        node_sets=[[0.0, 1.0, 3.0], [0.0, 0.5, 2.0]]
    ).Execute()
    grid.SetBlockIDFromFunction(lambda point, current: 1 if point.x < 1.0 else 2)
    grid.ExportToPVTU(output_base)
    barrier()

    roundtrip_grid = FromFileMeshGenerator(
        filename=f"{output_base}.pvtu",
        block_id_fieldname="Block",
        partitioner=LinearGraphPartitioner(),
    ).Execute()
    roundtrip_grid.SetOrthogonalBoundaries()
    volumes = roundtrip_grid.ComputeVolumePerBlockID()

    xs = MultiGroupXS()
    xs.LoadFromOpenSn("../../../assets/xs/pure_absorber_9g.xs")
    problem = DiscreteOrdinatesProblem(
        mesh=roundtrip_grid,
        num_groups=9,
        groupsets=[
            {
                "groups_from_to": (0, 8),
                "angular_quadrature": GLCProductQuadrature2DXY(
                    n_polar=4, n_azimuthal=8, scattering_order=0
                ),
                "angle_aggregation_type": "single",
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-12,
                "l_max_its": 100,
            }
        ],
        xs_map=[{"block_ids": [1, 2], "xs": xs}],
        volumetric_sources=[
            VolumetricSource(block_ids=[1, 2], group_strength=[1.0] * 9)
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

    domain = RPPLogicalVolume(infx=True, infy=True, infz=True)
    flux_error = 0.0
    for field_function in problem.GetScalarFluxFieldFunction():
        flux_error = max(
            flux_error,
            abs(field_extreme(field_function, "min", domain) - 1.0),
            abs(field_extreme(field_function, "max", domain) - 1.0),
        )

    balance = solver.ComputeBalanceTable()
    balance_error = abs(balance["production_rate"] - balance["absorption_rate"])

    if rank == 0:
        print(f"PVTU_BLOCK_1_VOLUME={volumes[1]:.12e}")
        print(f"PVTU_BLOCK_2_VOLUME={volumes[2]:.12e}")
        print(f"PVTU_FLUX_MAX_REL_ERROR={flux_error:.12e}")
        print(f"PVTU_BALANCE_ABS_ERROR={balance_error:.12e}")

    remove_output(output_base)
