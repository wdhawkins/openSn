#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D transient recurrence check on an unstructured extruded mesh (one-group absorber).

The problem is spatially uniform with reflecting boundaries, so each step should follow
the scalar theta-method recurrence:
  phi^{n+1} = ((1 - (1-theta)*a) * phi^n + b) / (1 + theta*a),
  a = rho * sigma_t * dt, b = q * dt.
The max scalar flux should match this recurrence at every step.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import ExtruderMeshGenerator, FromFileMeshGenerator, KBAGraphPartitioner
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, TransientSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume


def max_phi(problem):
    ff = problem.GetScalarFluxFieldFunction()[0]
    lv = RPPLogicalVolume(infx=True, infy=True, infz=True)
    ffi = FieldFunctionInterpolationVolume()
    ffi.SetOperationType("max")
    ffi.SetLogicalVolume(lv)
    ffi.AddFieldFunction(ff)
    ffi.Initialize()
    ffi.Execute()
    return ffi.GetValue()


if __name__ == "__main__":
    meshgen = ExtruderMeshGenerator(
        inputs=[FromFileMeshGenerator(filename="../../../../assets/mesh/triangle_mesh2x2_cuts.obj")],
        layers=[{"z": 0.5, "n": 1}, {"z": 1.0, "n": 1}],
        partitioner=KBAGraphPartitioner(nx=2, ny=2, xcuts=[0.0], ycuts=[0.0]),
    )
    grid = meshgen.Execute()
    grid.SetOrthogonalBoundaries()
    grid.SetUniformBlockID(0)

    sigma_t = 1.0
    rho = 1.5
    qval = 1.0
    dt = 0.05
    nsteps = 5

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t, 0.0, 1.0)

    pquad = GLCProductQuadrature3DXYZ(n_polar=2, n_azimuthal=4, scattering_order=0)

    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        time_dependent=True,
        groupsets=[{
            "groups_from_to": [0, 0],
            "angular_quadrature": pquad,
            "inner_linear_method": "petsc_gmres",
            "l_abs_tol": 1.0e-10,
            "l_max_its": 250,
        }],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[VolumetricSource(block_ids=[0], group_strength=[qval])],
        boundary_conditions=[
            {"name": "xmin", "type": "reflecting"},
            {"name": "xmax", "type": "reflecting"},
            {"name": "ymin", "type": "reflecting"},
            {"name": "ymax", "type": "reflecting"},
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
        ],
        density={"default_density": rho},
        options={
            "save_angular_flux": True,
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": False,
            "verbose_ags_iterations": False,
        },
    )

    solver = TransientSolver(problem=problem, dt=dt, theta=1.0, stop_time=nsteps * dt, initial_state="zero")
    solver.Initialize()

    a = rho * sigma_t * dt
    b = qval * dt
    phi_expected = 0.0
    max_rel = 0.0

    for _ in range(nsteps):
        solver.Advance()
        phi_expected = (phi_expected + b) / (1.0 + a)
        phi_num = max_phi(problem)
        rel = abs(phi_num - phi_expected) / phi_expected if phi_expected > 0.0 else 0.0
        max_rel = max(max_rel, rel)

    pass_flag = 1 if max_rel < 1.0e-6 else 0

    if rank == 0:
        print(f"DENSITY_3D_THETA_MAX_REL={max_rel:.8e}")
        print(f"DENSITY_3D_THETA_PASS {pass_flag}")
