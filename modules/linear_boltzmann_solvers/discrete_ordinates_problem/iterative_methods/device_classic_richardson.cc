// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/device_classic_richardson.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/sweep_wgs_context.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/convergence.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/iteration_logging.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/vecops/lbs_vecops.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "framework/utils/caliper_scopes.h"
#include "framework/utils/error.h"
#include "framework/utils/timer.h"
#include "caliper/cali.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <memory>

namespace opensn
{

namespace
{

bool
HasUnsupportedSourceFlags(const SourceFlags& scope,
                          const std::initializer_list<SourceType>& allowed_flags)
{
  const auto IsAllowed = [allowed_flags](const SourceType flag)
  {
    for (const auto allowed_flag : allowed_flags)
      if (flag == allowed_flag)
        return true;
    return false;
  };

  constexpr std::array<SourceType, 6> source_flags = {
    APPLY_FIXED_SOURCES,
    APPLY_WGS_SCATTER_SOURCES,
    APPLY_AGS_SCATTER_SOURCES,
    APPLY_WGS_FISSION_SOURCES,
    APPLY_AGS_FISSION_SOURCES,
    SUPPRESS_WG_SCATTER,
  };

  for (const auto flag : source_flags)
    if ((scope & flag) and not IsAllowed(flag))
      return true;

  return false;
}

bool
UseOrdinarySchedulerExperiment()
{
  const char* env_value =
    std::getenv("OPENSN_DEVICE_RICHARDSON_USE_ORDINARY_SCHEDULER"); // NOLINT(concurrency-mt-unsafe)
  if (env_value == nullptr)
    return false;

  const std::string value(env_value);
  return value == "1" or value == "true" or value == "TRUE" or value == "on" or value == "ON";
}

void
LogSweepProfileIfEnabled(const DeviceClassicRichardsonRuntime& runtime, const LBSGroupset& groupset)
{
  if (not runtime.ProfilingEnabled())
    return;

  const auto& profile = runtime.GetSweepProfile();
  if (profile.num_sweeps == 0)
    return;

  std::stringstream out;
  out << "device_classic_richardson profile groups [" << groupset.first_group << "-"
      << groupset.last_group << "] sweeps=" << profile.num_sweeps;
  AppendNumericField(out, "poll_s", profile.poll_seconds, Fixed(3));
  AppendNumericField(out, "incoming_copy_s", profile.incoming_copy_seconds, Fixed(3));
  AppendNumericField(out, "kernel_sync_s", profile.kernel_sync_seconds, Fixed(3));
  out << ", kernel_launches = " << profile.kernel_launch_count;
  out << ", ready_batches = " << profile.ready_batch_count;
  out << ", ready_avg = "
      << (profile.ready_batch_count > 0 ? profile.ready_batch_total_size / profile.ready_batch_count
                                        : 0);
  out << ", ready_max = " << profile.ready_batch_max_size;
  AppendNumericField(out, "send_s", profile.send_seconds, Fixed(3));
  AppendNumericField(out, "send_copy_s", profile.send_copy_seconds, Fixed(3));
  AppendNumericField(out, "send_dep_s", profile.send_dependency_seconds, Fixed(3));
  AppendNumericField(out, "send_mpi_s", profile.send_mpi_seconds, Fixed(3));
  out << ", send_msgs = " << profile.send_message_count;
  out << ", send_avg_doubles = "
      << (profile.send_message_count > 0 ? profile.send_total_doubles / profile.send_message_count
                                         : 0);
  out << ", send_max_doubles = " << profile.send_max_message_doubles;
  AppendNumericField(out, "finalize_s", profile.finalize_seconds, Fixed(3));
  AppendNumericField(out, "wait_s", profile.wait_seconds, Fixed(3));
  AppendNumericField(out, "post_s", profile.post_seconds, Fixed(3));
  if (mpi_comm.rank() == 0)
    std::cout << out.str() << std::endl;
}

} // namespace

void
RestoreHostPhiNewForOtherGroupsets(LBSProblem& lbs_problem,
                                   const LBSGroupset& groupset,
                                   const std::vector<double>& saved_phi_new)
{
  auto restored_phi_new = saved_phi_new;
  LBSVecOps::GSScopedCopyPrimarySTLvectors(
    lbs_problem, groupset, lbs_problem.GetPhiNewLocal(), restored_phi_new);
  lbs_problem.GetPhiNewLocal() = std::move(restored_phi_new);
}

DeviceClassicRichardson::DeviceClassicRichardson(
  const std::shared_ptr<SweepWGSContext>& gs_context_ptr, bool verbose)
  : LinearSystemSolver(gs_context_ptr->groupset.iterative_method, gs_context_ptr),
    sweep_context_(gs_context_ptr),
    runtime_(*gs_context_ptr),
    verbose_(verbose)
{
}

bool
DeviceClassicRichardson::SelectSourceBuildPath()
{
  use_fast_device_source_path_ = false;
  host_source_fallback_reason_.clear();

  const auto SetHostFallback = [this](const std::string& reason)
  {
    use_fast_device_source_path_ = false;
    host_source_fallback_reason_ = reason;
  };

  if (sweep_context_->groupset.force_host_source)
  {
    SetHostFallback("groupset.force_host_source=true");
    return false;
  }

  const auto& do_problem = sweep_context_->do_problem;
  if (do_problem.IsTimeDependent())
  {
    SetHostFallback("time-dependent sweeps are not supported by the fast device source path");
    return false;
  }

  if (do_problem.GetOptions().use_precursors)
  {
    SetHostFallback("delayed neutron precursors are not supported by the fast device source path");
    return false;
  }

  const auto lhs_has_unsupported_flags = HasUnsupportedSourceFlags(
    sweep_context_->lhs_src_scope, {APPLY_WGS_SCATTER_SOURCES, APPLY_WGS_FISSION_SOURCES});
  if (lhs_has_unsupported_flags)
  {
    SetHostFallback("lhs source scope contains flags outside {WGS scatter, WGS fission}");
    return false;
  }

  const auto rhs_has_unsupported_flags = HasUnsupportedSourceFlags(
    sweep_context_->rhs_src_scope,
    {APPLY_FIXED_SOURCES, APPLY_AGS_SCATTER_SOURCES, APPLY_AGS_FISSION_SOURCES});
  if (rhs_has_unsupported_flags)
  {
    SetHostFallback("rhs source scope contains flags outside {fixed, AGS scatter, AGS fission}");
    return false;
  }

  if (sweep_context_->lhs_src_scope & SUPPRESS_WG_SCATTER)
  {
    SetHostFallback("SUPPRESS_WG_SCATTER is not supported by the fast device source path");
    return false;
  }

  use_fast_device_source_path_ = true;
  return true;
}

void
DeviceClassicRichardson::Validate()
{
  const auto& do_problem = sweep_context_->do_problem;
  const auto& groupset = sweep_context_->groupset;

  OpenSnInvalidArgumentIf(not do_problem.UseGPUs(),
                          "device_classic_richardson requires use_gpus = true.");
  OpenSnInvalidArgumentIf(do_problem.GetSweepType() != "AAH",
                          "device_classic_richardson currently supports only GPU AAH sweeps.");
  OpenSnLogicalErrorIf(not sweep_context_,
                       "device_classic_richardson requires a sweep WGS context.");
  OpenSnInvalidArgumentIf(groupset.iterative_method !=
                            LinearSystemSolver::IterativeMethod::DEVICE_CLASSIC_RICHARDSON,
                          "DeviceClassicRichardson was selected for a non-device Richardson "
                          "groupset.");
  OpenSnInvalidArgumentIf(groupset.apply_wgdsa or groupset.apply_tgdsa,
                          "device_classic_richardson does not currently support DSA.");

  SelectSourceBuildPath();

  if (verbose_ and not UseFastDeviceSourcePath())
    log.Log() << "device_classic_richardson: using host source rebuild path for this groupset"
              << " (" << host_source_fallback_reason_ << ").";
}

bool
DeviceClassicRichardson::UseFastDeviceSourcePath() const
{
  return use_fast_device_source_path_;
}

void
DeviceClassicRichardson::InitializeLaggedState()
{
  auto& do_problem = sweep_context_->do_problem;
  auto& groupset = sweep_context_->groupset;

  saved_q_moments_local_ = do_problem.GetQMomentsLocal();

  if (UseFastDeviceSourcePath())
  {
    do_problem.SetQMomentsFrom(saved_q_moments_local_);
    sweep_context_->set_source_function(groupset,
                                        do_problem.GetQMomentsLocal(),
                                        do_problem.GetPhiOldLocal(),
                                        sweep_context_->rhs_src_scope);
    runtime_.CopySourceBaseToDevice();
  }

  runtime_.CopyPhiOldToDevice();
  do_problem.TransferDeviceBoundaryData(groupset.id, true, true);
  runtime_.UploadDelayedPsiToDevice();
}

void
DeviceClassicRichardson::BuildIterationSource()
{
  auto& do_problem = sweep_context_->do_problem;
  auto& groupset = sweep_context_->groupset;

  if (UseFastDeviceSourcePath())
  {
    runtime_.BuildSource(groupset,
                         sweep_context_->lhs_src_scope & APPLY_WGS_SCATTER_SOURCES,
                         sweep_context_->lhs_src_scope & APPLY_WGS_FISSION_SOURCES);
    return;
  }

  do_problem.SetQMomentsFrom(saved_q_moments_local_);
  sweep_context_->set_source_function(groupset,
                                      do_problem.GetQMomentsLocal(),
                                      do_problem.GetPhiOldLocal(),
                                      sweep_context_->lhs_src_scope |
                                        sweep_context_->rhs_src_scope);
  runtime_.CopySourceMomentsToDevice();
}

void
DeviceClassicRichardson::SyncLaggedStateToLatestIterate()
{
  runtime_.CopyDevicePhiNewToOld(sweep_context_->groupset);
  if (not UseFastDeviceSourcePath())
    runtime_.CopyPhiOldToHost();
}

void
DeviceClassicRichardson::Solve()
{
  CaliperPhaseScope cali_solve_phase("Solve", CaliperSolvePhaseDepth());
  CaliperRegionScope cali_wgs("WGS", CaliperWGSScopeDepth());
  CALI_CXX_MARK_SCOPE("DeviceClassicRichardson");

  sweep_context_->PreSetupCallback();
  sweep_context_->PreSolveCallback();
  Validate();

  auto& do_problem = sweep_context_->do_problem;
  auto& groupset = sweep_context_->groupset;
  const auto sweep_scope = sweep_context_->lhs_src_scope | sweep_context_->rhs_src_scope;
  const bool psi_check_active = groupset.angle_agg->GetNumDelayedAngularDOFs().first > 0;
  const bool use_ordinary_scheduler = UseOrdinarySchedulerExperiment();

  if (use_ordinary_scheduler)
    saved_q_moments_local_ = do_problem.GetQMomentsLocal();
  else
    InitializeLaggedState();

  std::vector<double> psi_old;
  std::vector<double> psi_new;
  if (psi_check_active and use_ordinary_scheduler)
    psi_old = groupset.angle_agg->GetOldDelayedAngularDOFsAsSTLVector();

  double pw_phi_change_prev = 1.0;
  double last_pw_phi_change = 0.0;
  bool converged = false;
  unsigned int num_iterations = 0;

  CALI_CXX_MARK_LOOP_BEGIN(wgs_iteration, "WGSIteration");
  for (unsigned int k = 0; k < groupset.max_iterations; ++k)
  {
    CALI_CXX_MARK_LOOP_ITERATION(wgs_iteration, k);

    num_iterations = k + 1;
    double pw_phi_change = 0.0;
    double pw_psi_change = 0.0;
    if (use_ordinary_scheduler)
    {
      do_problem.SetQMomentsFrom(saved_q_moments_local_);
      sweep_context_->set_source_function(
        groupset, do_problem.GetQMomentsLocal(), do_problem.GetPhiOldLocal(), sweep_scope);
      sweep_context_->ApplyInverseTransportOperator(sweep_scope);
      pw_phi_change = ComputePointwisePhiChange(do_problem, groupset.id);
      if (psi_check_active)
      {
        psi_new = groupset.angle_agg->GetNewDelayedAngularDOFsAsSTLVector();
        pw_psi_change = ComputePointwiseChange(psi_new, psi_old);
      }
    }
    else
    {
      BuildIterationSource();
      runtime_.ApplyInverseTransportOperator(sweep_scope);

      const auto metrics = runtime_.ComputeConvergenceMetrics(groupset);
      pw_phi_change = metrics.phi_change;
      pw_psi_change = metrics.psi_change;
    }
    last_pw_phi_change = pw_phi_change;
    const auto convergence = EvaluateRichardsonConvergence(
      pw_phi_change,
      pw_phi_change_prev,
      pw_psi_change,
      psi_check_active,
      groupset.residual_tolerance);
    pw_phi_change_prev = pw_phi_change;

    converged = convergence.converged;

    if (verbose_)
    {
      std::stringstream iter_stats;
      iter_stats << program_timer.GetTimeString() << " WGS groups [" << groupset.first_group << "-"
                 << groupset.last_group << "]"
                 << " iteration = " << k;
      AppendNumericField(iter_stats, "phi_change", pw_phi_change, Scientific(6));
      AppendNumericField(iter_stats, "psi_change", pw_psi_change, Scientific(6));
      AppendNumericField(iter_stats, "rho_est", convergence.rho, Fixed(4));

      if (converged)
        iter_stats << ", status = " << IterationStatusName(IterationStatus::CONVERGED);
      log.Log() << no_wrap << iter_stats.str();
    }

    if (not converged)
    {
      if (use_ordinary_scheduler)
      {
        LBSVecOps::GSScopedCopyPrimarySTLvectors(
          do_problem, groupset, PhiSTLOption::PHI_NEW, PhiSTLOption::PHI_OLD);
        if (psi_check_active)
          psi_old = psi_new;
      }
      else
        SyncLaggedStateToLatestIterate();
    }
    else
      break;
  }
  CALI_CXX_MARK_LOOP_END(wgs_iteration);

  sweep_context_->last_solve.num_iterations = num_iterations;
  sweep_context_->last_solve.status = IterationStatusFromSolve(converged, true);
  sweep_context_->last_solve.detail = converged ? "pointwise_converged" : "iteration_limit";
  sweep_context_->last_solve.metric_name = "phi_change";
  sweep_context_->last_solve.metric_value = last_pw_phi_change;

  if (verbose_ and (sweep_context_->last_solve.status == IterationStatus::FAILED or
                    sweep_context_->last_solve.status == IterationStatus::LIMIT))
    log.Log() << no_wrap << program_timer.GetTimeString() << " "
              << FormatIterationSummary("WGS groups [" + std::to_string(groupset.first_group) +
                                          "-" + std::to_string(groupset.last_group) + "]",
                                        sweep_context_->last_solve);

  do_problem.SetQMomentsFrom(saved_q_moments_local_);
  if (use_ordinary_scheduler)
  {
    if (psi_check_active)
      groupset.angle_agg->SetOldDelayedAngularDOFsFromSTLVector(
        groupset.angle_agg->GetNewDelayedAngularDOFsAsSTLVector());
    LBSVecOps::GSScopedCopyPrimarySTLvectors(
      do_problem, groupset, PhiSTLOption::PHI_NEW, PhiSTLOption::PHI_OLD);
  }
  else
  {
    const auto saved_phi_new = do_problem.GetPhiNewLocal();
    do_problem.CopyPhiAndOutflowBackToHost();
    RestoreHostPhiNewForOtherGroupsets(do_problem, groupset, saved_phi_new);
    do_problem.TransferDeviceBoundaryData(groupset.id, false, true);
    runtime_.DownloadDelayedPsiToHost();
    groupset.angle_agg->SetOldDelayedAngularDOFsFromSTLVector(
      groupset.angle_agg->GetNewDelayedAngularDOFsAsSTLVector());
    if (do_problem.SaveAngularFluxEnabled())
      runtime_.FinalizeAngularFluxes();
    LBSVecOps::GSScopedCopyPrimarySTLvectors(
      do_problem, groupset, PhiSTLOption::PHI_NEW, PhiSTLOption::PHI_OLD);
  }

  LogSweepProfileIfEnabled(runtime_, groupset);
  sweep_context_->PostSolveCallback();
}

} // namespace opensn
