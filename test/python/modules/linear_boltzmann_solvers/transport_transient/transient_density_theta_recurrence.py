#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transient theta-scheme recurrence check in a reflected one-group absorber/source slab.

With uniform source q and no scattering, each time step should satisfy:
  phi^{n+1} = ((1 - (1-theta)*a) * phi^n + b) / (1 + theta*a),
  a = rho*sigma_t*dt, b = q*dt.
Expected answer:
  transport solution matches this recurrence for both theta=1 (BE) and theta=0.5 (CN).
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLProductQuadrature1DSlab
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


def run_case(theta):
    nodes = [i / 30.0 for i in range(31)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes]).Execute()
    grid.SetUniformBlockID(0)

    sigma_t = 1.0
    rho = 1.3
    qval = 1.5
    dt = 0.05
    nsteps = 6

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t, 0.0, 1.0)

    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        time_dependent=True,
        groupsets=[{
            "groups_from_to": [0, 0],
            "angular_quadrature": GLProductQuadrature1DSlab(n_polar=6, scattering_order=0),
            "inner_linear_method": "petsc_gmres",
            "l_abs_tol": 1.0e-10,
            "l_max_its": 250,
        }],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[VolumetricSource(block_ids=[0], group_strength=[qval])],
        boundary_conditions=[
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

    solver = TransientSolver(problem=problem, dt=dt, theta=theta, stop_time=nsteps * dt, initial_state="zero")
    solver.Initialize()

    a = rho * sigma_t * dt
    b = qval * dt
    phi_expected = 0.0
    max_rel = 0.0

    for _ in range(nsteps):
        solver.Advance()
        phi_expected = ((1.0 - (1.0 - theta) * a) * phi_expected + b) / (1.0 + theta * a)

        phi_num = max_phi(problem)
        rel = abs(phi_num - phi_expected) / phi_expected if phi_expected > 0.0 else 0.0
        max_rel = max(max_rel, rel)

    return max_rel


if __name__ == "__main__":
    rel_be = run_case(1.0)
    rel_cn = run_case(0.5)
    pass_flag = 1 if (rel_be < 1.0e-6 and rel_cn < 1.0e-6) else 0

    if rank == 0:
        print(f"DENSITY_THETA_REL_BE={rel_be:.8e}")
        print(f"DENSITY_THETA_REL_CN={rel_cn:.8e}")
        print(f"DENSITY_THETA_PASS {pass_flag}")
