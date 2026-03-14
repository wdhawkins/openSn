#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt-alpha density-scaling check in a supercritical prompt-only transient.

After converging a critical state, XS are switched to a prompt-supercritical set and
alpha is estimated from fission-production growth:
  alpha = ln(F(t1)/F(t0)) / (t1-t0).
For this setup, effective prompt kinetics scale approximately linearly with uniform rho.
Expected answer:
  alpha(rho=2) / alpha(rho=1) ~= 2.
"""

import math
import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.solver import (
        DiscreteOrdinatesProblem,
        PowerIterationKEigenSolver,
        TransientSolver,
    )
    from pyopensn.aquad import GLProductQuadrature1DSlab
    from pyopensn.xs import MultiGroupXS
    from pyopensn.mesh import OrthogonalMeshGenerator


def run_case(rho):
    nodes = [float(i) for i in range(9)]
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes])
    grid = meshgen.Execute()
    grid.SetUniformBlockID(0)

    xs_crit = MultiGroupXS()
    xs_crit.LoadFromOpenSn("xs1g_prompt_crit.cxs")

    xs_super = MultiGroupXS()
    xs_super.LoadFromOpenSn("xs1g_prompt_super.cxs")

    pquad = GLProductQuadrature1DSlab(n_polar=8, scattering_order=0)

    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[{
            "groups_from_to": [0, 0],
            "angular_quadrature": pquad,
            "inner_linear_method": "classic_richardson",
            "l_abs_tol": 1.0e-8,
            "l_max_its": 250,
        }],
        xs_map=[{"block_ids": [0], "xs": xs_crit}],
        boundary_conditions=[
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
        ],
        density={"default_density": rho},
        options={
            "save_angular_flux": True,
            "use_precursors": False,
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": False,
            "verbose_ags_iterations": False,
        },
    )

    keigen = PowerIterationKEigenSolver(problem=problem, max_iters=200, k_tol=1.0e-10)
    keigen.Initialize()
    keigen.Execute()

    problem.SetTimeDependentMode()

    dt = 1.0e-2
    nsteps = 10
    solver = TransientSolver(problem=problem, dt=dt, theta=1.0, stop_time=nsteps * dt, initial_state="existing")
    solver.Initialize()

    problem.SetXSMap(xs_map=[{"block_ids": [0], "xs": xs_super}])

    fp0 = problem.ComputeFissionProduction("new")
    for _ in range(nsteps):
        solver.Advance()
    fp1 = problem.ComputeFissionProduction("new")

    alpha = math.log(fp1 / fp0) / (nsteps * dt)
    return alpha


if __name__ == "__main__":
    alpha1 = run_case(1.0)
    alpha2 = run_case(2.0)

    ratio = alpha2 / alpha1 if alpha1 != 0.0 else 0.0
    ratio_err = abs(ratio - 2.0) / 2.0

    pass_flag = 1 if ratio_err < 2.0e-2 else 0

    if rank == 0:
        print(f"DENSITY_ALPHA1={alpha1:.8e}")
        print(f"DENSITY_ALPHA2={alpha2:.8e}")
        print(f"DENSITY_ALPHA_RATIO={ratio:.8e}")
        print(f"DENSITY_ALPHA_RATIO_ERR={ratio_err:.8e}")
        print(f"DENSITY_ALPHA_PASS {pass_flag}")
