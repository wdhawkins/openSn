// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/spmd_threadpool.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/source_functions/source_flags.h"
#include <cstddef>
#include <utility>
#include <vector>

namespace opensn
{

class DiscreteOrdinatesProblem;
class LBSGroupset;
struct SweepWGSContext;
class AAHDSweepChunk;
class AAHD_AngleSet;

struct DeviceRichardsonConvergenceMetrics
{
  double phi_change = 0.0;
  double psi_change = 0.0;
};

class DeviceClassicRichardsonRuntime
{
public:
  explicit DeviceClassicRichardsonRuntime(SweepWGSContext& sweep_context);

  void CopyPhiOldToDevice();
  void CopyPhiOldToHost();
  void CopySourceMomentsToDevice();
  void CopyDevicePhiNewToOld(const LBSGroupset& groupset);
  void CopySourceBaseToDevice();
  void BuildSource(const LBSGroupset& groupset, bool apply_wgs_scatter, bool apply_wgs_fission);
  double ComputeLocalPhiChange(const LBSGroupset& groupset) const;
  double ComputeGlobalPhiChange(const LBSGroupset& groupset) const;
  DeviceRichardsonConvergenceMetrics ComputeConvergenceMetrics(const LBSGroupset& groupset) const;
  void UploadDelayedPsiToDevice();
  void DownloadDelayedPsiToHost();
  void ApplyInverseTransportOperator(SourceFlags scope);
  double GetLastDelayedPsiRelativeChange() const { return last_delayed_psi_relative_change_; }
  void FinalizeAngularFluxes();

private:
  void InitializeSweepResources();
  void PrepareSweep(bool use_boundary_source, bool zero_incoming_delayed_psi);
  void ExecuteSweepPass(bool final_download);
  void AccumulateSweepStats(double sweep_time_seconds, size_t sweep_passes);

private:
  SweepWGSContext* sweep_context_ = nullptr;
  DiscreteOrdinatesProblem* problem_ = nullptr;
  LBSGroupset* groupset_ = nullptr;
  AAHDSweepChunk* sweep_chunk_ = nullptr;
  std::vector<AAHD_AngleSet*> angle_sets_;
  SPMD_ThreadPool pool_;
  std::vector<std::size_t> execution_order_;
  size_t last_sweep_pass_count_ = 0;
  double last_local_delayed_psi_relative_change_ = 0.0;
  double last_delayed_psi_relative_change_ = 0.0;
};

} // namespace opensn
