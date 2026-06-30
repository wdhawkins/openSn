// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/sweep_wgs_context.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/preconditioning/lbs_shell_operations.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/vecops/lbs_vecops.h"
#include "framework/math/petsc_utils/petsc_utils.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "framework/utils/timer.h"
#include "framework/utils/caliper_scopes.h"
#include "caliper/cali.h"
#include <petscksp.h>

namespace opensn
{

using PCShellPtr = PetscErrorCode (*)(PC, Vec, Vec);
using namespace std::chrono;

namespace
{

inline SchedulingAlgorithm
GetSchedulingAlgorithm(const std::string& sweep_type, bool use_gpu)
{
  if (sweep_type == "AAH")
  {
    if (use_gpu)
      return SchedulingAlgorithm::ALL_AT_ONCE;
    else
      return SchedulingAlgorithm::DEPTH_OF_GRAPH;
  }
  else if (sweep_type == "CBC")
  {
    if (use_gpu)
      return SchedulingAlgorithm::ASYNC_FIFO;
    else
      return SchedulingAlgorithm::FIRST_IN_FIRST_OUT;
  }
  else
    throw std::runtime_error("Unsupported sweep scheduling algorithm: " + sweep_type + "\n");
}

} // namespace

SweepWGSContext::SweepWGSContext(DiscreteOrdinatesProblem& do_problem,
                                 LBSGroupset& groupset,
                                 const SetSourceFunction& set_source_function,
                                 SourceFlags lhs_scope,
                                 SourceFlags rhs_scope,
                                 bool log_info,
                                 std::shared_ptr<SweepChunk> swp_chnk)
  : WGSContext(do_problem, groupset, set_source_function, lhs_scope, rhs_scope, log_info),
    sweep_chunk(std::move(swp_chnk)),
    sweep_scheduler(GetSchedulingAlgorithm(do_problem.GetSweepType(), do_problem.UseGPUs()),
                    *groupset.angle_agg,
                    *sweep_chunk)
{
  if (not groupset.fixed_point_sweep)
  {
    const auto delayed_local = groupset.angle_agg->GetLocalDelayedAngularDOFBreakdown();
    AngleAggregation::DelayedAngularDOFBreakdown delayed_global;
    mpi_comm.all_reduce(delayed_local.boundary, delayed_global.boundary, mpi::op::sum<size_t>());
    mpi_comm.all_reduce(delayed_local.local, delayed_global.local, mpi::op::sum<size_t>());
    mpi_comm.all_reduce(delayed_local.nonlocal, delayed_global.nonlocal, mpi::op::sum<size_t>());
    mpi_comm.all_reduce(delayed_local.ab, delayed_global.ab, mpi::op::sum<size_t>());
    mpi_comm.all_reduce(delayed_local.promoted, delayed_global.promoted, mpi::op::sum<size_t>());

    log.Log0() << no_wrap << program_timer.GetTimeString() << " Groupset " << groupset.id
               << " delayed angular dofs local: total=" << delayed_local.Total()
               << ", boundary=" << delayed_local.boundary << ", local=" << delayed_local.local
               << ", nonlocal=" << delayed_local.nonlocal << ", ab=" << delayed_local.ab
               << ", promoted=" << delayed_local.promoted;

    log.Log0() << no_wrap << program_timer.GetTimeString() << " Groupset " << groupset.id
               << " delayed angular dofs global: total=" << delayed_global.Total()
               << ", boundary=" << delayed_global.boundary << ", local=" << delayed_global.local
               << ", nonlocal=" << delayed_global.nonlocal << ", ab=" << delayed_global.ab
               << ", promoted=" << delayed_global.promoted;
  }
}

void
SweepWGSContext::RebuildAngularFluxFromConvergedPhi(bool include_rhs_time_term,
                                                    bool zero_incoming_delayed_psi)
{
  CALI_CXX_MARK_SCOPE("AngularFluxReconstruction");

  auto scope = lhs_src_scope | rhs_src_scope;
  if (zero_incoming_delayed_psi)
    scope |= ZERO_INCOMING_DELAYED_PSI;
  {
    CALI_CXX_MARK_SCOPE("Source");
    set_source_function(
      groupset, do_problem.GetQMomentsLocal(), do_problem.GetPhiOldLocal(), scope);
  }

  sweep_chunk->IncludeRHSTimeTerm(include_rhs_time_term);
  if (groupset.fixed_point_sweep and groupset.fixed_point_sweep_defer_angular_flux_pullback)
    sweep_scheduler.ForceFinalAngularFluxPullback();
  ApplyInverseTransportOperator(scope);
  LBSVecOps::GSScopedCopyPrimarySTLvectors(
    do_problem, groupset, PhiSTLOption::PHI_NEW, PhiSTLOption::PHI_OLD);
}

void
SweepWGSContext::SetPreconditioner(KSP& solver)
{

  auto& ksp = solver;

  PC pc = nullptr;
  OpenSnPETScCall(KSPGetPC(ksp, &pc));

  if (groupset.apply_wgdsa or groupset.apply_tgdsa)
  {
    OpenSnPETScCall(PCSetType(pc, PCSHELL));
    OpenSnPETScCall(PCShellSetApply(pc, (PCShellPtr)WGDSA_TGDSA_PreConditionerMult));
    OpenSnPETScCall(PCShellSetContext(pc, &(*this)));
  }

  OpenSnPETScCall(KSPSetPCSide(ksp, PC_LEFT));
  OpenSnPETScCall(KSPSetUp(ksp));
}

std::pair<int64_t, int64_t>
SweepWGSContext::GetSystemSize()
{

  const size_t local_node_count = do_problem.GetLocalNodeCount();
  const auto global_node_count = do_problem.GetGlobalNodeCount();
  const auto num_moments = do_problem.GetNumMoments();

  const auto groupset_numgrps = groupset.GetNumGroups();
  const auto num_delayed_psi_info =
    groupset.fixed_point_sweep ? std::pair<size_t, size_t>{0, 0}
                               : groupset.angle_agg->GetNumDelayedAngularDOFs();
  const size_t local_size =
    local_node_count * num_moments * groupset_numgrps + num_delayed_psi_info.first;
  const auto global_size =
    global_node_count * num_moments * groupset_numgrps + num_delayed_psi_info.second;

  return {static_cast<int64_t>(local_size), static_cast<int64_t>(global_size)};
}

void
SweepWGSContext::PreSolveCallback()
{
}

void
SweepWGSContext::ApplyInverseTransportOperator(SourceFlags scope)
{
  CaliperRegionScope cali_sweep_region("Sweep", CaliperSweepScopeDepth());

  ++counter_applications_of_inv_op;

  // Sweep
  const bool use_bndry_source_flag =
    (scope & APPLY_FIXED_SOURCES) and (not do_problem.GetOptions().use_src_moments);
  const bool zero_incoming_delayed_psi = (scope & ZERO_INCOMING_DELAYED_PSI);
  do_problem.ZeroOutflowBalanceVars(groupset);
  sweep_scheduler.PrepareForSweep(use_bndry_source_flag, zero_incoming_delayed_psi);

  high_resolution_clock::time_point sweep_start = high_resolution_clock::now();
  sweep_scheduler.Sweep();
  high_resolution_clock::time_point sweep_end = high_resolution_clock::now();

  auto sweep_time =
    static_cast<double>(duration_cast<nanoseconds>(sweep_end - sweep_start).count()) / 1.0e+9;
  sweep_stats_.total_sweep_time += sweep_time;
  ++sweep_stats_.num_sweeps;
  sweep_stats_.num_sweep_passes += sweep_scheduler.GetLastSweepPassCount();
}

void
SweepWGSContext::PostSolveCallback()
{

  // Perform final sweep with converged phi and delayed psi dofs. This step is necessary for
  // Krylov methods to recover the actual solution (this includes all of the PETSc methods
  // currently used in OpenSn).
  const bool needs_deferred_angular_flux_reconstruction =
    groupset.fixed_point_sweep and groupset.fixed_point_sweep_defer_angular_flux_pullback and
    do_problem.SaveAngularFluxEnabled();
  if (needs_deferred_angular_flux_reconstruction or
      groupset.iterative_method == LinearSystemSolver::IterativeMethod::PETSC_GMRES or
      groupset.iterative_method == LinearSystemSolver::IterativeMethod::PETSC_BICGSTAB or
      (groupset.iterative_method == LinearSystemSolver::IterativeMethod::PETSC_RICHARDSON and
       groupset.max_iterations > 1))
  {
    RebuildAngularFluxFromConvergedPhi(sweep_chunk->IsTimeDependent());
  }
}

} // namespace opensn
