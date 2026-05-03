// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

namespace opensn
{

class DiffusionMIPSolver;
class DiscreteOrdinatesProblem;
class SweepResidualEvaluator;

/** Problem-level all-groups diffusion acceleration applied after an AGS iteration. */
class UDSAAcceleration
{
public:
  explicit UDSAAcceleration(DiscreteOrdinatesProblem& do_problem);

  void Initialize();
  void Apply();

private:
  void CopyScalarFlux(const std::vector<double>& phi_in, std::vector<double>& phi0) const;
  void ProjectScalarFlux(const std::vector<double>& phi0, std::vector<double>& phi_out) const;
  void BuildFixedSourceMoments(std::vector<double>& source_moments) const;
  void BuildBoundarySource(std::vector<double>& boundary_source) const;
  void BuildDiffusionSource(const std::vector<double>& phi0, std::vector<double>& q0) const;
  void BuildCurrentCorrection(std::vector<double>& correction);
  double ComputeL2Change(const std::vector<double>& current,
                         const std::vector<double>& previous) const;

  DiscreteOrdinatesProblem& do_problem_;
  std::shared_ptr<DiffusionMIPSolver> diffusion_solver_;
  std::shared_ptr<SweepResidualEvaluator> sweep_residual_evaluator_;
  std::vector<double> phi0_;
  std::vector<double> phi0_old_;
  std::vector<double> q0_;
  std::vector<double> fixed_source_moments_;
  std::vector<double> boundary_source_;
  std::vector<double> current_correction_;
};

} // namespace opensn
