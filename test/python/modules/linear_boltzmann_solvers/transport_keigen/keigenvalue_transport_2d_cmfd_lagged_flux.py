#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D 7-group pin lattice with CMFD and one transport sweep per update, with and without the lagged
angular flux correction.

The model is the 3x3 C5G7 lattice of keigenvalue_transport_2d_cmfd_overlay.py, with reflecting
boundaries on all sides. Opposing reflecting boundaries lag their incoming angular flux by one
sweep. With one sweep per update, CMFD rescales the scalar flux but the next sweep still reads the
unscaled lagged boundary flux, which slows convergence. The lagged correction scales that flux by
the same coarse-cell ratios as the scalar flux.

The reference is the converged unaccelerated power iteration on the same discretization. Both CMFD
runs must converge to it within the eigenvalue tolerances. The corrected run must also need
clearly fewer sweeps: if the corrected lagged flux does not reach the next sweep, both runs take
the same iterations and the sweep counts are equal.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem, PowerIterationKEigenSolver
    from pyopensn.solver import CMFDAcceleration


if __name__ == "__main__":

    pitch = 1.26
    num_pins = 3
    cells_per_pin = 6
    fuel_cells = 4
    num_groups = 7
    k_tolerance = 1.0e-8

    xs_water = MultiGroupXS()
    xs_water.LoadFromOpenSn("../../../../assets/xs/c5g7/XS_water.xs")
    xs_uo2 = MultiGroupXS()
    xs_uo2.LoadFromOpenSn("../../../../assets/xs/c5g7/XS_UO2.xs")
    xs_mox = MultiGroupXS()
    xs_mox.LoadFromOpenSn("../../../../assets/xs/c5g7/XS_8_7pMOX.xs")

    def make_problem(inner_linear_method, max_its, abs_tol):
        dx = pitch / cells_per_pin
        nodes = [i * dx for i in range(num_pins * cells_per_pin + 1)]
        grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes]).Execute()
        grid.SetUniformBlockID(0)
        margin = 0.5 * (cells_per_pin - fuel_cells) * dx
        for i in range(num_pins):
            for j in range(num_pins):
                if (i, j) == (1, 1):
                    continue
                fuel_block = 2 if (i, j) == (num_pins - 1, num_pins - 1) else 1
                grid.SetBlockIDFromLogicalVolume(
                    RPPLogicalVolume(
                        xmin=i * pitch + margin,
                        xmax=(i + 1) * pitch - margin,
                        ymin=j * pitch + margin,
                        ymax=(j + 1) * pitch - margin,
                        infz=True,
                    ),
                    fuel_block,
                    True,
                )

        return DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=num_groups,
            groupsets=[
                {
                    "groups_from_to": (0, num_groups - 1),
                    "angular_quadrature": GLCProductQuadrature2DXY(
                        n_polar=2, n_azimuthal=8, scattering_order=0
                    ),
                    "inner_linear_method": inner_linear_method,
                    "l_max_its": max_its,
                    "l_abs_tol": abs_tol,
                }
            ],
            xs_map=[
                {"block_ids": [0], "xs": xs_water},
                {"block_ids": [1], "xs": xs_uo2},
                {"block_ids": [2], "xs": xs_mox},
            ],
            boundary_conditions=[
                {"name": name, "type": "reflecting"}
                for name in ["xmin", "xmax", "ymin", "ymax"]
            ],
            options={
                "verbose_inner_iterations": False,
                "verbose_outer_iterations": False,
            },
        )

    def run_power_iteration():
        problem = make_problem("petsc_gmres", 50, 1.0e-11)
        solver = PowerIterationKEigenSolver(problem=problem, max_iters=2000, k_tol=1.0e-10)
        solver.Initialize()
        solver.Execute()
        return solver.GetEigenvalue()

    def run_cmfd(lagged):
        problem = make_problem("classic_richardson", 1, 1.0e-12)
        cmfd = CMFDAcceleration(
            problem=problem,
            coarse_mesh="reactor_lattice",
            lattice_origin=[0.0, 0.0],
            lattice_pitch=[pitch, pitch],
            lattice_shape=[num_pins, num_pins],
            update_wgs_max_its=1,
            update_wgs_abs_tol=1.0e-12,
            coarse_solver_policy="direct",
            correct_lagged_angular_flux=lagged,
        )
        solver = PowerIterationKEigenSolver(
            problem=problem, acceleration=cmfd, max_iters=1000, k_tol=k_tolerance
        )
        solver.Initialize()
        solver.Execute()
        return solver.GetEigenvalue(), solver.GetNumSweeps(), int(solver.IsConverged())

    pi_k = run_power_iteration()
    plain_k, plain_sweeps, plain_converged = run_cmfd(False)
    lagged_k, lagged_sweeps, lagged_converged = run_cmfd(True)
    sweep_ratio = lagged_sweeps / plain_sweeps

    if rank == 0:
        print(f"Python PI k-eigenvalue: {pi_k:.10f}")
        print(f"Python CMFD k-eigenvalue: {plain_k:.10f}")
        print(f"Python CMFD sweeps: {plain_sweeps}")
        print(f"Python CMFD converged: {plain_converged}")
        print(f"Python lagged CMFD k-eigenvalue: {lagged_k:.10f}")
        print(f"Python lagged CMFD sweeps: {lagged_sweeps}")
        print(f"Python lagged CMFD converged: {lagged_converged}")
        print(f"Python lagged CMFD-PI k-difference: {abs(lagged_k - pi_k):.3e}")
        print(f"Python lagged/plain sweep ratio: {sweep_ratio:.4f}")
