// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/linear_solver/petsc_linear_solver.h"
#include "framework/runtime.h"

namespace opensn
{

int
LinearSolverMatrixAction(Mat matrix, Vec vector, Vec action)
{
  LinearSolverContext* context;
  MatShellGetContext(matrix, &context);

  context->MatrixAction(matrix, vector, action);

  return 0;
}

PETScLinearSolver::PETScLinearSolver(const std::string& petsc_iterative_method,
                                     std::shared_ptr<LinearSolverContext> context_ptr)
  : LinearSolver(context_ptr),
    A_(nullptr),
    b_(nullptr),
    x_(nullptr),
    ksp_(nullptr),
    num_local_dofs_(0),
    num_global_dofs_(0),
    petsc_iterative_method_(petsc_iterative_method),
    system_set_(false),
    suppress_kspsolve_(false)
{
}

PETScLinearSolver::~PETScLinearSolver()
{
  VecDestroy(&x_);
  VecDestroy(&b_);
  KSPDestroy(&ksp_);
}

void
PETScLinearSolver::ApplyToleranceOptions()
{
  KSPSetTolerances(ksp_,
                   tolerance_options.residual_relative,
                   tolerance_options.residual_absolute,
                   tolerance_options.residual_divergence,
                   tolerance_options.maximum_iterations);
}

void
PETScLinearSolver::PreSetupCallback()
{
}

void
PETScLinearSolver::SetOptions()
{
}

void
PETScLinearSolver::SetSolverContext()
{
  KSPSetApplicationContext(ksp_, &(*context_ptr_));
}

void
PETScLinearSolver::SetConvergenceTest()
{
  KSPSetConvergenceTest(ksp_, &KSPConvergedDefault, nullptr, nullptr);
}

void
PETScLinearSolver::SetMonitor()
{
}

void
PETScLinearSolver::SetPreconditioner()
{
}

void
PETScLinearSolver::PostSetupCallback()
{
}

void
PETScLinearSolver::Setup()
{
  if (IsSystemSet())
    return;

  PreSetupCallback();
  KSPCreate(opensn::mpi_comm, &ksp_);
  KSPSetType(ksp_, petsc_iterative_method_.c_str());
  ApplyToleranceOptions();
  if (petsc_iterative_method_ == "gmres")
  {
    KSPGMRESSetRestart(ksp_, tolerance_options.gmres_restart_interval);
    KSPGMRESSetBreakdownTolerance(ksp_, tolerance_options.gmres_breakdown_tolerance);
  }
  KSPSetInitialGuessNonzero(ksp_, PETSC_FALSE);
  SetOptions();
  SetSolverContext();
  SetConvergenceTest();
  SetMonitor();
  SetSystemSize();
  SetSystem();
  SetPreconditioner();
  PostSetupCallback();
  system_set_ = true;
}

void
PETScLinearSolver::PreSolveCallback()
{
}

void
PETScLinearSolver::PostSolveCallback()
{
}

void
PETScLinearSolver::Solve()
{
  PreSolveCallback();
  SetInitialGuess();
  SetRHS();
  if (not suppress_kspsolve_)
    KSPSolve(ksp_, b_, x_);
  PostSolveCallback();
}

} // namespace opensn
