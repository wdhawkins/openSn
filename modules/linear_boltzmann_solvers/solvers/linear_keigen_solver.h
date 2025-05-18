// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/physics/solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/linear_keigen_solver.h"
#include <petscsnes.h>

namespace opensn
{

class LinearKEigenSolver : public Solver
{
private:
  std::shared_ptr<LBSProblem> lbs_problem_;
  std::shared_ptr<LinearKEigenAGSContext> context_;
  LinearKEigenAGSSolver solver_;

  bool reset_phi0_;

public:
  explicit LinearKEigenSolver(const InputParameters& params);

  void Initialize() override;
  void Execute() override;

  /// Return the current k-eigenvalue
  double GetEigenvalue() const;

public:
  static InputParameters GetInputParameters();
  static std::shared_ptr<LinearKEigenSolver> Create(const ParameterBlock& params);
};

} // namespace opensn
