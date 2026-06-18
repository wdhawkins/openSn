// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/wgs_context.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/device_classic_richardson_runtime.h"
#include "framework/math/linear_solver/linear_system_solver.h"
#include <memory>
#include <string>
#include <vector>

namespace opensn
{

struct SweepWGSContext;

class DeviceClassicRichardson : public LinearSystemSolver
{
public:
  explicit DeviceClassicRichardson(const std::shared_ptr<SweepWGSContext>& gs_context_ptr,
                                   bool verbose);

  ~DeviceClassicRichardson() override = default;

  void Solve() override;

private:
  void SelectSourceBuildPath();
  bool UseFastDeviceSourcePath() const;
  void Validate();
  void InitializeLaggedState();
  void BuildIterationSource();
  void SyncLaggedStateToLatestIterate();
  double ComputePhiChange() const;
  double ComputePsiChange() const;

private:
  std::shared_ptr<SweepWGSContext> sweep_context_;
  DeviceClassicRichardsonRuntime runtime_;
  bool verbose_;
  bool use_fast_device_source_path_ = false;
  std::string host_source_fallback_reason_;
  std::vector<double> saved_q_moments_local_;
};

} // namespace opensn
