// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/solvers/linear_keigen_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/wgs_context.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/ags_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_vecops.h"
#include "framework/math/petsc_utils/petsc_utils.h"
#include "framework/logging/log.h"
#include "framework/utils/timer.h"
#include "framework/object_factory.h"
#include <slepceps.h>

namespace opensn
{

namespace
{

// Shell matrix multiplication: y = A * x
PetscErrorCode ShellMult(Mat M, Vec x, Vec y)
{
  void* ctx = nullptr;
  MatShellGetContext(M, &ctx);

  auto* linear_ctx = static_cast<LinearKEigenAGSContext*>(ctx);
  auto& lbs_problem = linear_ctx->lbs_problem;
  //auto ags_solver = lbs_problem->GetAGSSolver();
  const auto& phi_old_local = lbs_problem->GetPhiOldLocal();
  auto& q_moments_local = lbs_problem->GetQMomentsLocal();
  auto active_set_source_function = lbs_problem->GetActiveSetSourceFunction();

  LBSVecOps::SetPrimarySTLvectorFromMultiGSPETScVec(*lbs_problem, linear_ctx->groupset_ids, x, PhiSTLOption::PHI_OLD);

  Set(q_moments_local, 0.0);
  for (auto& groupset : lbs_problem->GetGroupsets())
  {
    active_set_source_function(groupset,
		               q_moments_local,
			       phi_old_local,
			       APPLY_AGS_FISSION_SOURCES | APPLY_WGS_FISSION_SOURCES);
  }
  //ags_solver->Solve();
  for (auto& solver : lbs_problem->GetWGSSolvers())
  {
    solver->Setup();
    solver->Solve();
  }

  LBSVecOps::SetMultiGSPETScVecFromPrimarySTLvector(*lbs_problem, linear_ctx->groupset_ids, y, PhiSTLOption::PHI_NEW);

  return 0;
}

} 

static PetscErrorCode LinearKEigenMonitor(EPS eps,
		                          PetscInt its,
					  PetscInt nconv,
					  PetscScalar *eigr,
					  PetscScalar *eigi,
					  PetscReal *errest,
					  PetscInt nest,
					  void *ctx)
{
  std::stringstream iter_info;
  iter_info << program_timer.GetTimeString() << " SLEPc EPS Iteration " << std::setw(5) << its;
  if (nconv > 0) 
    iter_info << " Residual " << std::setw(11) << errest[0];
  else
    iter_info << " No converged eigenpairs";
  log.Log() << iter_info.str();

  return 0;
}

void LinearKEigenAGSSolver::PreSetupCallback()
{
  auto ctx = std::dynamic_pointer_cast<LinearKEigenAGSContext>(context_ptr_);
  ctx->groupset_ids.clear();
  for (auto& gs : ctx->lbs_problem->GetGroupsets())
    ctx->groupset_ids.push_back(gs.id);
}

void LinearKEigenAGSSolver::SetMonitor()
{
}

void LinearKEigenAGSSolver::SetSystemSize()
{
  auto ctx = std::dynamic_pointer_cast<LinearKEigenAGSContext>(context_ptr_);
  auto sizes = ctx->lbs_problem->GetNumPhiIterativeUnknowns();
  num_local_dofs_ = static_cast<int64_t>(sizes.first);
  num_global_dofs_ = static_cast<int64_t>(sizes.second);
}

void LinearKEigenAGSSolver::SetSystem()
{
  // Create shell matrix A
  MatCreateShell(PETSC_COMM_WORLD,
                 (PetscInt)num_local_dofs_, (PetscInt)num_local_dofs_,
                 (PetscInt)num_global_dofs_, (PetscInt)num_global_dofs_,
                 context_ptr_.get(), &A_);
  MatShellSetOperation(A_, MATOP_MULT, (void(*)(void))ShellMult);
  // Create x
  x_ = CreateVector(num_local_dofs_, num_global_dofs_);
}

void LinearKEigenAGSSolver::SetInitialGuess()
{
  auto ctx = std::dynamic_pointer_cast<LinearKEigenAGSContext>(context_ptr_);
  LBSVecOps::SetMultiGSPETScVecFromPrimarySTLvector(*ctx->lbs_problem, ctx->groupset_ids, x_, PhiSTLOption::PHI_OLD);
}

void LinearKEigenAGSSolver::Solve()
{
  PreSolveCallback();

  EPSSetOperators(eps_, A_, nullptr);
  EPSSetProblemType(eps_, EPS_NHEP);
  EPSSetWhichEigenpairs(eps_, EPS_LARGEST_MAGNITUDE);
  EPSSetTolerances(eps_, tolerance_options.residual_absolute, tolerance_options.maximum_iterations);
  EPSSetType(eps_, eps_type_.c_str());
  EPSSetInitialSpace(eps_, /*n=*/1, &x_);
  EPSMonitorSet(eps_, LinearKEigenMonitor, nullptr, nullptr);
  EPSSetFromOptions(eps_);
  EPSSolve(eps_);

  // Extract dominant eigenvalue
  PetscScalar eigenval;
  EPSGetEigenpair(eps_, 0, &eigenval, nullptr, nullptr, nullptr);
  /*PetscInt nconv;
  EPSGetConverged(eps_, &nconv);
  EPSGetEigenpair(eps_, nconv-1, &eigenval, nullptr, nullptr, nullptr); */
  auto ctx = std::dynamic_pointer_cast<LinearKEigenAGSContext>(context_ptr_);
  ctx->eigenvalue = PetscRealPart(eigenval);

  // Extract eigenvector directly into x_
  EPSGetEigenvector(eps_, 0, x_, nullptr);

  PostSolveCallback();
}

void LinearKEigenAGSSolver::PreSolveCallback()
{
  auto ctx = std::dynamic_pointer_cast<LinearKEigenAGSContext>(context_ptr_);
  auto& lbs_problem = ctx->lbs_problem;

  for (auto& wgs_solver : lbs_problem->GetWGSSolvers())
  {
    auto context = wgs_solver->GetContext();
    auto wgs_context = std::dynamic_pointer_cast<WGSContext>(context);
    wgs_context->lhs_src_scope.Unset(APPLY_WGS_FISSION_SOURCES); // lhs_scope
    wgs_context->rhs_src_scope.Unset(APPLY_AGS_FISSION_SOURCES); // rhs_scope
  }
}

void LinearKEigenAGSSolver::PostSolveCallback()
{
  auto ctx = std::dynamic_pointer_cast<LinearKEigenAGSContext>(context_ptr_);
  LBSVecOps::SetPrimarySTLvectorFromMultiGSPETScVec(*ctx->lbs_problem, ctx->groupset_ids, x_, PhiSTLOption::PHI_NEW);
  log.Log() << "Linear k-eigenvalue: " << ctx->eigenvalue << "\n";
}

} // namespace opensn
