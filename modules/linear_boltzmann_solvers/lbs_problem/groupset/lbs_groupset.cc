// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/quadratures/angular/angular_quadrature.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "framework/logging/log.h"
#include "framework/object_factory.h"
#include "framework/runtime.h"
#include <fstream>

namespace opensn
{

OpenSnRegisterObjectParametersOnlyInNamespace(lbs, LBSGroupset);

InputParameters
LBSGroupset::GetInputParameters()
{
  InputParameters params;

  params.SetGeneralDescription("Input Parameters for groupsets.");

  // Groupsets
  params.AddRequiredParameterArray("groups_from_to",
                                   "The first and last group id this groupset operates on, e.g. A "
                                   "4 group problem <TT>groups_from_to= {0, 3}</TT>");
  // Anglesets
  params.AddOptionalParameter<std::shared_ptr<AngularQuadrature>>(
    "angular_quadrature", nullptr, "A handle to an angular quadrature");
  params.AddOptionalParameter(
    "angle_aggregation_type", "polar", "The angle aggregation method to use during sweeping");
  params.AddOptionalParameter("angle_aggregation_num_sets",
                              96,
                              "Target number of angle sets for s4_coalesced angle aggregation. "
                              "The final count can be larger only when conservative splitting "
                              "options are enabled.");
  params.AddOptionalParameter(
    "angle_aggregation_target_angles_per_set",
    0,
    "For s4_coalesced angle aggregation, target this many angles per angle set. "
    "A value of 0 disables this option. When positive, this takes precedence over "
    "angle_aggregation_num_sets and derives the set count as ceil(num_angles / target).");
  params.AddOptionalParameter("angle_aggregation_split_partition_faces",
                              false,
                              "For s4_coalesced angle aggregation, split angle sets when angles "
                              "disagree on inter-rank partition face orientation. This is a "
                              "conservative fallback; by default, disagreement is handled with "
                              "promoted cross-rank delayed angular unknowns.");
  params.AddOptionalParameter(
    "fixed_point_sweep",
    false,
    "For device-richardson with coalesced angle sets, converge lagged angular fluxes by fixed-point "
    "sweep passes instead of adding them to the iterative unknown vector.");
  params.AddOptionalParameter("fixed_point_sweep_max_iterations",
                              20,
                              "Maximum number of fixed-point sweep passes for a fixed source.");
  params.AddOptionalParameter("fixed_point_sweep_tolerance",
                              1.0e-6,
                              "Relative lagged angular-flux tolerance for fixed-point sweeps.");
  params.AddOptionalParameter(
    "fixed_point_sweep_defer_angular_flux_pullback",
    false,
    "For fixed-point GPU sweeps with saved angular fluxes, defer saved angular-flux pullback "
    "until the final reconstruction sweep.");

  // Iterative method
  params.AddOptionalParameter("inner_linear_method",
                              "petsc_richardson",
                              "The iterative method to use for inner linear solves");
  params.AddOptionalParameter(
    "l_abs_tol", 1.0e-6, "Inner linear solver residual absolute tolerance");
  params.AddOptionalParameter("l_max_its", 200, "Inner linear solver maximum iterations");
  params.AddOptionalParameter("gmres_restart_interval",
                              30,
                              "If this inner linear solver is gmres, sets the number of "
                              "iterations before a restart occurs.");
  params.AddOptionalParameter(
    "allow_cycles", true, "Flag indicating whether cycles are to be allowed or not");

  // WG DSA options
  params.AddOptionalParameter("apply_wgdsa",
                              false,
                              "Flag to turn on within-group Diffusion Synthetic Acceleration for "
                              "this groupset");
  params.AddOptionalParameter(
    "wgdsa_l_abs_tol", 1.0e-4, "Within-group DSA linear absolute tolerance");
  params.AddOptionalParameter("wgdsa_l_max_its", 30, "Within-group DSA linear maximum iterations");
  params.AddOptionalParameter("wgdsa_solver_policy",
                              "auto",
                              "Within-group DSA diffusion solver policy. auto uses a serial "
                              "direct PETSc LU solve below wgdsa_direct_solve_threshold global "
                              "unknowns and iterative CG/BoomerAMG otherwise. direct, iterative, "
                              "and petsc_options force a specific path.");
  params.AddOptionalParameter("wgdsa_direct_solve_threshold",
                              20000,
                              "Maximum global unknown count for the automatic WGDSA direct PETSc "
                              "LU path.");
  params.AddOptionalParameter(
    "wgdsa_verbose", false, "If true, WGDSA routines will print verbosely");
  params.AddOptionalParameter("wgdsa_petsc_options", "", "PETSc options to pass to WGDSA solver");

  // TG DSA options
  params.AddOptionalParameter(
    "apply_tgdsa", false, "Flag to turn on Two-Grid Acceleration for this groupset");
  params.AddOptionalParameter("tgdsa_l_abs_tol", 1.0e-4, "Two-Grid DSA linear absolute tolerance");
  params.AddOptionalParameter("tgdsa_l_max_its", 30, "Two-Grid DSA linear maximum iterations");
  params.AddOptionalParameter("tgdsa_solver_policy",
                              "auto",
                              "Two-grid DSA diffusion solver policy. auto uses a serial direct "
                              "PETSc LU solve below tgdsa_direct_solve_threshold global unknowns "
                              "and iterative CG/BoomerAMG otherwise. direct, iterative, and "
                              "petsc_options force a specific path.");
  params.AddOptionalParameter("tgdsa_direct_solve_threshold",
                              20000,
                              "Maximum global unknown count for the automatic TGDSA direct PETSc "
                              "LU path.");
  params.AddOptionalParameter(
    "tgdsa_verbose", false, "If true, TGDSA routines will print verbosely");
  params.AddOptionalParameter("tgdsa_petsc_options", "", "PETSc options to pass to TGDSA solver");

  // Constraints
  params.ConstrainParameterRange(
    "angle_aggregation_type",
    AllowableRangeList::New({"polar", "single", "azimuthal", "s4_coalesced"}));
  params.ConstrainParameterRange("angle_aggregation_num_sets", AllowableRangeLowLimit::New(1));
  params.ConstrainParameterRange("angle_aggregation_target_angles_per_set",
                                 AllowableRangeLowLimit::New(0));
  params.ConstrainParameterRange("fixed_point_sweep_max_iterations",
                                 AllowableRangeLowLimit::New(1));
  params.ConstrainParameterRange("fixed_point_sweep_tolerance",
                                 AllowableRangeLowLimit::New(0.0));
  params.ConstrainParameterRange(
    "inner_linear_method",
    AllowableRangeList::New(
      {"classic_richardson", "petsc_richardson", "petsc_gmres", "petsc_bicgstab"}));
  params.ConstrainParameterRange("l_abs_tol", AllowableRangeLowLimit::New(1.0e-18));
  params.ConstrainParameterRange("l_max_its", AllowableRangeLowLimit::New(0));
  params.ConstrainParameterRange("gmres_restart_interval", AllowableRangeLowLimit::New(1));
  params.ConstrainParameterRange(
    "wgdsa_solver_policy",
    AllowableRangeList::New({"auto", "direct", "iterative", "petsc_options"}));
  params.ConstrainParameterRange(
    "tgdsa_solver_policy",
    AllowableRangeList::New({"auto", "direct", "iterative", "petsc_options"}));
  params.ConstrainParameterRange("wgdsa_direct_solve_threshold", AllowableRangeLowLimit::New(1));
  params.ConstrainParameterRange("tgdsa_direct_solve_threshold", AllowableRangeLowLimit::New(1));

  return params;
}

// The functionality provided by Init() should really be accomplished via the member initializer
// list. But, delegating constructors won't work here, and repeating the initialization is worse,
// in my opinion, than an init method.
void
LBSGroupset::Init(int aid)
{
  id = aid;
  first_group = 0;
  last_group = 0;
  size = 0;
  quadrature = nullptr;
  angle_agg = nullptr;
  angle_aggregation_num_sets = 96;
  angle_aggregation_target_angles_per_set = 0;
  angle_aggregation_split_partition_faces = false;
  fixed_point_sweep = false;
  fixed_point_sweep_max_iterations = 20;
  fixed_point_sweep_tolerance = 1.0e-6;
  fixed_point_sweep_defer_angular_flux_pullback = false;
  iterative_method = LinearSystemSolver::IterativeMethod::PETSC_RICHARDSON;
  angleagg_method = AngleAggregationType::POLAR;
  residual_tolerance = 1.0e-6;
  max_iterations = 200;
  gmres_restart_intvl = 30;
  allow_cycles = false;
  apply_wgdsa = false;
  apply_tgdsa = false;
  wgdsa_max_iters = 30;
  tgdsa_max_iters = 30;
  wgdsa_tol = 1.0e-4;
  tgdsa_tol = 1.0e-4;
  wgdsa_solver_policy = "auto";
  tgdsa_solver_policy = "auto";
  wgdsa_direct_solve_threshold = 20000;
  tgdsa_direct_solve_threshold = 20000;
  wgdsa_verbose = false;
  tgdsa_verbose = false;
  wgdsa_solver = nullptr;
  tgdsa_solver = nullptr;
}

LBSGroupset::LBSGroupset() // NOLINT(cppcoreguidelines-pro-type-member-init)
{
  Init(-1);
};

LBSGroupset::LBSGroupset(int id) // NOLINT(cppcoreguidelines-pro-type-member-init)
{
  Init(id);
}

LBSGroupset::LBSGroupset( // NOLINT(cppcoreguidelines-pro-type-member-init)
  const InputParameters& params,
  const int id,
  const LBSProblem& lbs_problem)
{
  Init(id);

  // Add groups
  const auto groups_from_to = params.GetParamVectorValue<unsigned int>("groups_from_to");
  OpenSnInvalidArgumentIf(groups_from_to.size() != 2,
                          "Parameter \"groups_from_to\" can only have 2 entries");

  first_group = groups_from_to[0];
  last_group = groups_from_to[1];
  OpenSnInvalidArgumentIf(last_group < first_group,
                          "\"to\" field is less than the \"from\" field.");
  size = last_group - first_group + 1;

  // Add quadrature
  quadrature = params.GetSharedPtrParam<AngularQuadrature>("angular_quadrature", false);

  // Angle aggregation
  const auto angle_agg_typestr = params.GetParamValue<std::string>("angle_aggregation_type");
  if (angle_agg_typestr == "polar")
    angleagg_method = AngleAggregationType::POLAR;
  else if (angle_agg_typestr == "single")
    angleagg_method = AngleAggregationType::SINGLE;
  else if (angle_agg_typestr == "azimuthal")
    angleagg_method = AngleAggregationType::AZIMUTHAL;
  else if (angle_agg_typestr == "s4_coalesced")
    angleagg_method = AngleAggregationType::S4_COALESCED;

  angle_aggregation_num_sets = params.GetParamValue<int>("angle_aggregation_num_sets");
  angle_aggregation_target_angles_per_set =
    params.GetParamValue<int>("angle_aggregation_target_angles_per_set");
  angle_aggregation_split_partition_faces =
    params.GetParamValue<bool>("angle_aggregation_split_partition_faces");
  fixed_point_sweep = params.GetParamValue<bool>("fixed_point_sweep");
  fixed_point_sweep_max_iterations =
    params.GetParamValue<unsigned int>("fixed_point_sweep_max_iterations");
  fixed_point_sweep_tolerance = params.GetParamValue<double>("fixed_point_sweep_tolerance");
  fixed_point_sweep_defer_angular_flux_pullback =
    params.GetParamValue<bool>("fixed_point_sweep_defer_angular_flux_pullback");

  // Inner solver
  const auto inner_linear_method = params.GetParamValue<std::string>("inner_linear_method");
  if (inner_linear_method == "classic_richardson")
    iterative_method = LinearSystemSolver::IterativeMethod::CLASSIC_RICHARDSON;
  else if (inner_linear_method == "petsc_richardson")
    iterative_method = LinearSystemSolver::IterativeMethod::PETSC_RICHARDSON;
  else if (inner_linear_method == "petsc_gmres")
    iterative_method = LinearSystemSolver::IterativeMethod::PETSC_GMRES;
  else if (inner_linear_method == "petsc_bicgstab")
    iterative_method = LinearSystemSolver::IterativeMethod::PETSC_BICGSTAB;

  gmres_restart_intvl = params.GetParamValue<int>("gmres_restart_interval");
  allow_cycles = params.GetParamValue<bool>("allow_cycles");
  residual_tolerance = params.GetParamValue<double>("l_abs_tol");
  max_iterations = params.GetParamValue<unsigned int>("l_max_its");

  // DSA
  apply_wgdsa = params.GetParamValue<bool>("apply_wgdsa");
  apply_tgdsa = params.GetParamValue<bool>("apply_tgdsa");

  wgdsa_tol = params.GetParamValue<double>("wgdsa_l_abs_tol");
  tgdsa_tol = params.GetParamValue<double>("tgdsa_l_abs_tol");

  wgdsa_max_iters = params.GetParamValue<unsigned int>("wgdsa_l_max_its");
  tgdsa_max_iters = params.GetParamValue<unsigned int>("tgdsa_l_max_its");

  wgdsa_solver_policy = params.GetParamValue<std::string>("wgdsa_solver_policy");
  tgdsa_solver_policy = params.GetParamValue<std::string>("tgdsa_solver_policy");
  wgdsa_direct_solve_threshold = params.GetParamValue<unsigned int>("wgdsa_direct_solve_threshold");
  tgdsa_direct_solve_threshold = params.GetParamValue<unsigned int>("tgdsa_direct_solve_threshold");

  wgdsa_verbose = params.GetParamValue<bool>("wgdsa_verbose");
  tgdsa_verbose = params.GetParamValue<bool>("tgdsa_verbose");

  wgdsa_string = params.GetParamValue<std::string>("wgdsa_petsc_options");
  tgdsa_string = params.GetParamValue<std::string>("tgdsa_petsc_options");
}

#ifndef __OPENSN_WITH_GPU__
void
LBSGroupset::InitializeQuadratureCarrier()
{
}

void
LBSGroupset::ResetQuadratureCarrier()
{
}
#endif // __OPENSN_WITH_GPU__

LBSGroupset::~LBSGroupset()
{
  ResetQuadratureCarrier();
}

unsigned int
LBSGroupset::GetNumGroups() const
{
  return size;
}

} // namespace opensn
