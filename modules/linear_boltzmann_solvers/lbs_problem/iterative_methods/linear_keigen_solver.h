// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/linear_solver/slepc_linear_eigen_solver.h"
#include "framework/math/linear_solver/linear_solver_context.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include <memory>
#include <vector>

namespace opensn
{

/// Context for the linear k-eigenvalue solve
struct LinearKEigenAGSContext : public LinearEigenContext
{
  std::shared_ptr<LBSProblem> lbs_problem;
  std::vector<int> groupset_ids;

  explicit LinearKEigenAGSContext(std::shared_ptr<LBSProblem> lbs)
    : lbs_problem(std::move(lbs))
  {}

  ~LinearKEigenAGSContext() override = default;
};

/// A PETSc/SLEPc shell-based linear k-eigenvalue solver
class LinearKEigenAGSSolver : public SLEPcLinearEigenSolver
{
public:
  explicit LinearKEigenAGSSolver(std::shared_ptr<LinearKEigenAGSContext> context)
    : SLEPcLinearEigenSolver(IterativeMethod::SLEPC_SOLVER, std::move(context))
  {}

  ~LinearKEigenAGSSolver() override = default;

  void Solve() override;

protected:
  void PreSetupCallback() override;
  void SetMonitor() override;
  void SetSystemSize() override;
  void SetSystem() override;
  void SetInitialGuess() override;
  void PostSolveCallback() override;
  void PreSolveCallback() override;
};

} // namespace opensn
