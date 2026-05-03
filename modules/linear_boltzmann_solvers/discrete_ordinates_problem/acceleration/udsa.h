// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace opensn
{

class DiffusionMIPSolver;
class DiscreteOrdinatesProblem;
class UnknownManager;
class SweepResidualEvaluator;

/** Reusable all-groups diffusion acceleration machinery for UDSA. */
class UDSADiffusionAcceleration
{
public:
  explicit UDSADiffusionAcceleration(DiscreteOrdinatesProblem& do_problem);

  void Initialize();

  bool IsInitialized() const { return static_cast<bool>(diffusion_solver_); }
  const UnknownManager& GetUnknownManager() const;
  void SetSolverOptions(double residual_tolerance,
                        unsigned int max_iters,
                        bool verbose,
                        const std::string& petsc_options);

  void CopyScalarFlux(const std::vector<double>& phi_in, std::vector<double>& phi0) const;
  void ProjectScalarFlux(const std::vector<double>& phi0, std::vector<double>& phi_out) const;
  void BuildFixedSourceMoments(std::vector<double>& source_moments) const;
  void BuildBoundarySource(std::vector<double>& boundary_source) const;
  void BuildDiffusionSource(const std::vector<double>& phi0,
                            const std::vector<double>& fixed_source_moments,
                            std::vector<double>& q0) const;
  void BuildCurrentCorrection(const std::vector<double>& phi0,
                              const std::vector<double>& boundary_source,
                              std::vector<double>& correction);
  void AssembleRHS(const std::vector<double>& q0,
                   const std::vector<double>& boundary_source,
                   const std::vector<double>& current_correction);
  void Solve(std::vector<double>& phi0, bool use_initial_guess);
  double ComputeL2Change(const std::vector<double>& current,
                         const std::vector<double>& previous) const;
  size_t GetNumLocalDOFs() const;

  DiscreteOrdinatesProblem& do_problem_;
  std::shared_ptr<DiffusionMIPSolver> diffusion_solver_;
  std::shared_ptr<SweepResidualEvaluator> sweep_residual_evaluator_;
};

/** Problem-level all-groups diffusion acceleration applied after an AGS iteration. */
class UDSAAcceleration
{
public:
  explicit UDSAAcceleration(DiscreteOrdinatesProblem& do_problem);

  void Initialize();
  void Apply();

private:
  DiscreteOrdinatesProblem& do_problem_;
  UDSADiffusionAcceleration diffusion_acceleration_;
  std::vector<double> phi0_;
  std::vector<double> phi0_old_;
  std::vector<double> q0_;
  std::vector<double> fixed_source_moments_;
  std::vector<double> boundary_source_;
  std::vector<double> current_correction_;
};

} // namespace opensn
