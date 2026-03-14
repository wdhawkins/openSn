#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transient pure-decay recurrence check from an existing steady state.

A steady source solution is first computed, then the source is removed and one
Backward-Euler step is taken for:
  dphi/dt + rho*sigma_t*phi = 0.
Expected update:
  phi^{n+1} = phi^n / (1 + rho*sigma_t*dt).
This test compares phi^{n+1} to that exact BE recurrence value.
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
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver, TransientSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume


def avg_phi(problem):
    ff = problem.GetScalarFluxFieldFunction()[0]
    lv = RPPLogicalVolume(infx=True, infy=True, infz=True)
    ffi = FieldFunctionInterpolationVolume()
    ffi.SetOperationType("avg")
    ffi.SetLogicalVolume(lv)
    ffi.AddFieldFunction(ff)
    ffi.Initialize()
    ffi.Execute()
    return ffi.GetValue()


if __name__ == "__main__":
    nodes = [i / 30.0 for i in range(31)]
    grid = OrthogonalMeshGenerator(node_sets=[nodes]).Execute()
    grid.SetUniformBlockID(0)

    sigma_t = 1.0
    rho = 1.4
    dt = 0.04
    nsteps = 1

    xs = MultiGroupXS()
    xs.CreateSimpleOneGroup(sigma_t, 0.0, 1.0)

    source_on = VolumetricSource(block_ids=[0], group_strength=[1.8])

    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{
            "groups_from_to": [0, 0],
            "angular_quadrature": GLProductQuadrature1DSlab(n_polar=6, scattering_order=0),
            "inner_linear_method": "petsc_gmres",
            "l_abs_tol": 1.0e-10,
            "l_max_its": 250,
        }],
        xs_map=[{"block_ids": [0], "xs": xs}],
        volumetric_sources=[source_on],
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

    steady = SteadyStateSourceSolver(problem=problem)
    steady.Initialize()
    steady.Execute()

    phi0 = avg_phi(problem)

    problem.SetTimeDependentMode()
    problem.SetVolumetricSources(clear_volumetric_sources=True)

    solver = TransientSolver(problem=problem, dt=dt, theta=1.0, stop_time=nsteps * dt, initial_state="existing")
    solver.Initialize()

    a = rho * sigma_t * dt
    solver.Advance()
    phi_expected = phi0 / (1.0 + a)
    phi_num = avg_phi(problem)
    max_rel = abs(phi_num - phi_expected) / phi_expected if phi_expected > 0.0 else 0.0

    pass_flag = 1 if max_rel < 1.0e-6 else 0

    if rank == 0:
        print(f"DENSITY_DECAY_PHI0={phi0:.8e}")
        print(f"DENSITY_DECAY_PHI1={phi_num:.8e}")
        print(f"DENSITY_DECAY_EXP1={phi_expected:.8e}")
        print(f"DENSITY_DECAY_MAX_REL={max_rel:.8e}")
        print(f"DENSITY_DECAY_PASS {pass_flag}")
