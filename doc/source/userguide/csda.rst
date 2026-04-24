================================
CSDA Charged-Particle Transport
================================

OpenSn's CSDA mode adds continuous-slowing-down charged-particle transport to a
steady-state discrete-ordinates solve. It is intended for electron and positron
CEPXS-BFP data that includes stopping power, charge deposition, and energy
deposition data.

The CSDA implementation augments the angular sweep with a groupwise energy-loss
term. It also carries enough terminal charged-particle information to report
particle balance and CSDA-adjusted deposition field functions.

Basic Workflow
==============

The usual CSDA workflow is:

1. Load CEPXS data with ``csda_format=True``.
2. Enable CSDA on a Cartesian :py:class:`pyopensn.solver.DiscreteOrdinatesProblem`.
3. Put each contiguous charged-particle group block in a single groupset.
4. Solve the problem to a tight enough convergence tolerance for balance work.
5. Use CSDA field functions and :py:meth:`ComputeBalanceTable` to verify the
   result.

Example:

.. code-block:: python

   from pyopensn.xs import MultiGroupXS
   from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver

   xs = MultiGroupXS()
   xs.LoadFromCEPXS("plastic_csda.bxslib", material_id=0, csda_format=True)

   problem = DiscreteOrdinatesProblem(
       mesh=mesh,
       num_groups=xs.num_groups,
       groupsets=[
           {
               "groups_from_to": (0, xs.num_groups - 1),
               "angular_quadrature": quadrature,
               "inner_linear_method": "gmres",
               "l_abs_tol": 1.0e-10,
           },
       ],
       xs_map=[{"block_ids": [0], "xs": xs}],
       boundary_conditions=boundary_conditions,
       volumetric_sources=volumetric_sources,
       options={"csda_enabled": True},
       sweep_type="AAH",
   )

   solver = SteadyStateSourceSolver(problem=problem, compute_balance=True)
   solver.Initialize()
   solver.Execute()

   balance = solver.ComputeBalanceTable()
   edep = problem.CreateFieldFunction("edep", "csda_energy_deposition")
   cdep = problem.CreateFieldFunction("cdep", "csda_charge_deposition")

CEPXS CSDA Data
===============

Use :py:meth:`pyopensn.xs.MultiGroupXS.LoadFromCEPXS` for CEPXS-BFP binary
libraries:

.. code-block:: python

   xs.LoadFromCEPXS("material.bxslib", material_id=0, csda_format=True)

The ``csda_format`` flag selects the CEPXS row convention used by OpenSn's CSDA
import path. With ``csda_format=True``, OpenSn imports:

* transfer data for the transport solve,
* ``charge_deposition`` as a named custom one-dimensional cross section,
* energy deposition data,
* stopping power data used by the CSDA sweep.

Materials without stopping power remain ordinary transport materials. Materials
with stopping power must provide one stopping-power value per energy group and
energy bounds for every group.

Problem Requirements
====================

CSDA is enabled with the problem option ``csda_enabled``:

.. code-block:: python

   problem = DiscreteOrdinatesProblem(
       ...,
       options={"csda_enabled": True},
   )

Current restrictions are:

* CSDA is supported for forward steady-state Cartesian
  :py:class:`pyopensn.solver.DiscreteOrdinatesProblem` solves.
* It is not supported for adjoint mode.
* It is not supported for time-dependent mode.
* It is not supported with GPU sweeps.
* It is not supported with ``sweep_type="CBC"``.
* It is not supported by
  :py:class:`pyopensn.solver.DiscreteOrdinatesCurvilinearProblem`.
* Each contiguous charged-particle group range, identified by nonzero stopping
  power, must be wholly contained in one groupset.

The last restriction is important. For example, if nonzero stopping power occurs
in groups 20 through 90, a groupset boundary may not split that range. Neutral
or photon groups outside the charged range may be placed in separate groupsets.

Derived Field Functions
=======================

CSDA adds several names to
:py:meth:`pyopensn.solver.LBSProblem.CreateFieldFunction`.

``energy_deposition``
---------------------

When CSDA is disabled, this is the raw imported energy-deposition cross section
weighted by scalar flux. When ``csda_enabled=True``, this name is an alias for
``csda_energy_deposition``.

``csda_energy_deposition``
--------------------------

This field includes the raw imported energy-deposition response plus the CSDA
stopping-power energy-loss contribution from charged groups. It also includes
the terminal cutoff-energy contribution associated with the terminal
charged-particle current.

``charge_deposition``
---------------------

This is the raw CEPXS ``charge_deposition`` custom cross section weighted by
scalar flux.

``csda_charge_deposition``
--------------------------

This field includes the raw ``charge_deposition`` response plus the CSDA
terminal charge-deposition correction.

``csda_charge_deposition_term``
-------------------------------

This field contains only the terminal CSDA charge-deposition correction,
projected as a normal derived nodal field.

``csda_charge_deposition_term_cellavg``
---------------------------------------

This field contains only the terminal CSDA charge-deposition correction as a
piecewise cell-average field.

Balance Table Entries
=====================

:py:meth:`ComputeBalanceTable` returns the ordinary balance entries:

* ``absorption_rate``
* ``production_rate``
* ``inflow_rate``
* ``outflow_rate``
* ``balance``

For CSDA charged-particle solves it also returns:

* ``csda_charge_deposition_rate``: the integrated terminal charge-deposition
  rate,
* ``csda_particle_balance``: ``production_rate + inflow_rate`` minus
  ``absorption_rate + outflow_rate + csda_charge_deposition_rate``,
* ``csda_energy_deposition_rate``: the integrated CSDA energy-deposition rate.

For steady-state source solves, no additional solver normalization is applied to
these values. Eigenvalue solvers apply the same ``1 / k_eff`` production scaling
used by their ordinary balance table entries.

Convergence and Verification
============================

CSDA balance is convergence-sensitive. The reported ``csda_particle_balance``
should decrease as the transport solve is converged more tightly. If the balance
does not improve when the linear tolerance is tightened, check that:

* the charged-particle group block is not split across groupsets,
* the solve actually converged to the requested tolerance,
* angular fluxes and balance tallies are computed after the final scalar
  iterate has been rebuilt,
* the CEPXS library was loaded with ``csda_format=True``.

For charged-particle CEPXS problems, GMRES or BiCGSTAB is usually the safer
inner linear method. Classic Richardson can be used when its pointwise iteration
is known to converge for the case being run, but a non-contracting Richardson
iteration can produce a poor solution even if the input data and CSDA balance
formulas are otherwise correct.
