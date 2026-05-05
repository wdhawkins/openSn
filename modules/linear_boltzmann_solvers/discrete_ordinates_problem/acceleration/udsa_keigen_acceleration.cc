// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/udsa_keigen_acceleration.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/solvers/pi_keigen_solver.h"
#include "modules/diffusion/diffusion_mip_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/compute/lbs_compute.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/iteration_logging.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/source_functions/source_flags.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/vecops/lbs_vecops.h"
#include "framework/object_factory.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "framework/utils/timer.h"
#include "caliper/cali.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace opensn
{

namespace
{

double
GlobalL2Norm(const std::vector<double>& values)
{
  double local_sum = 0.0;
  for (const double value : values)
    local_sum += value * value;

  double global_sum = 0.0;
  mpi_comm.all_reduce(&local_sum, 1, &global_sum, mpi::op::sum<double>());
  return std::sqrt(global_sum);
}

} // namespace

OpenSnRegisterObjectInNamespace(lbs, UDSAKEigenAcceleration);

InputParameters
UDSAKEigenAcceleration::GetInputParameters()
{
  auto params = DiscreteOrdinatesKEigenAcceleration::GetInputParameters();
  params.ChangeExistingParamToOptional("name", "UDSAKEigenAcceleration");
  params.AddOptionalParameter("use_transport_eigenvalue",
                              true,
                              "If true, UDSA corrects the fission-source shape but returns the "
                              "transport power-iteration eigenvalue. If false, UDSA returns the "
                              "low-order diffusion eigenvalue estimate.");
  return params;
}

std::shared_ptr<UDSAKEigenAcceleration>
UDSAKEigenAcceleration::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<UDSAKEigenAcceleration>("lbs::UDSAKEigenAcceleration", params);
}

UDSAKEigenAcceleration::UDSAKEigenAcceleration(const InputParameters& params)
  : DiscreteOrdinatesKEigenAcceleration(params),
    use_transport_eigenvalue_(params.GetParamValue<bool>("use_transport_eigenvalue")),
    diffusion_acceleration_(do_problem_,
                            UDSADiffusionAcceleration::ScatterCouplingMode::Operator)
{
}

void
UDSAKEigenAcceleration::Initialize()
{
  CALI_CXX_MARK_SCOPE("UDSAKEigenAcceleration::Initialize");

  for (const auto& [bid, bc] : do_problem_.GetBoundaryDefinitions())
    if ((bc.first == LBSBoundaryType::ISOTROPIC) or (bc.first == LBSBoundaryType::ARBITRARY))
      throw std::logic_error("UDSA k-eigenvalue acceleration only supports vacuum and reflective "
                             "boundaries.");

  diffusion_acceleration_.Initialize();
  const auto petsc_options =
    petsc_options_.empty()
      ? "-" + do_problem_.GetName() + "_UDSAksp_type gmres -" + do_problem_.GetName() +
          "_UDSApc_type hypre"
      : petsc_options_;
  diffusion_acceleration_.SetSolverOptions(l_abs_tol_, max_iters_, verbose_, petsc_options);

  const auto num_local_dofs = diffusion_acceleration_.GetNumLocalDOFs();
  phi0_.assign(num_local_dofs, 0.0);
  phi0_ell_.assign(num_local_dofs, 0.0);
  phi0_star_.assign(num_local_dofs, 0.0);
  phi0_m_.assign(num_local_dofs, 0.0);
  phi0_kp1_.assign(num_local_dofs, 0.0);
  q0_.assign(num_local_dofs, 0.0);
  sweep_source_.assign(num_local_dofs, 0.0);
  source_correction_.assign(num_local_dofs, 0.0);
  boundary_source_.assign(num_local_dofs, 0.0);
  current_correction_.assign(num_local_dofs, 0.0);
  operator_phi_.assign(num_local_dofs, 0.0);
  phi_temp_.assign(phi_new_local_.size(), 0.0);
}

void
UDSAKEigenAcceleration::PreExecute()
{
}

void
UDSAKEigenAcceleration::PrePowerIteration()
{
  production_old_ = ComputeFissionProduction(do_problem_, phi_old_local_);
  diffusion_acceleration_.CopyScalarFlux(phi_old_local_, phi0_ell_);
}

void
UDSAKEigenAcceleration::BuildFissionSource(const std::vector<double>& phi0,
                                           const double lambda,
                                           std::vector<double>& q0) const
{
  std::vector<double> phi(do_problem_.GetPhiNewLocal().size(), 0.0);
  std::vector<double> source_moments(phi.size(), 0.0);
  diffusion_acceleration_.ProjectScalarFlux(phi0, phi);
  const auto set_source_function = do_problem_.GetActiveSetSourceFunction();
  set_source_function(front_gs_,
                      source_moments,
                      phi,
                      APPLY_AGS_FISSION_SOURCES | APPLY_WGS_FISSION_SOURCES);
  diffusion_acceleration_.CopyScalarFlux(source_moments, q0);
  for (auto& value : q0)
    value /= lambda;
}

double
UDSAKEigenAcceleration::PostPowerIteration()
{
  CALI_CXX_MARK_SCOPE("UDSAKEigenAcceleration::PostPowerIteration");

  diffusion_acceleration_.CopyScalarFlux(phi_new_local_, phi0_star_);
  const double lambda = solver_->GetEigenvalue();
  const double transport_production = ComputeFissionProduction(do_problem_, phi_new_local_);
  OpenSnLogicalErrorIf(production_old_ == 0.0,
                       "UDSA k-eigenvalue acceleration encountered zero previous production.");
  const double transport_lambda = transport_production / production_old_ * lambda;

  BuildFissionSource(phi0_ell_, lambda, sweep_source_);
  std::fill(boundary_source_.begin(), boundary_source_.end(), 0.0);
  std::fill(current_correction_.begin(), current_correction_.end(), 0.0);
  std::fill(source_correction_.begin(), source_correction_.end(), 0.0);
  phi0_m_ = phi0_star_;
  double production_m = transport_production;
  double lambda_m = transport_lambda;

  double change = 0.0;
  unsigned int iter = 0;
  unsigned int iteration_count = 0;
  for (; iter < static_cast<unsigned int>(std::max(pi_max_its_, 1)); ++iter)
  {
    iteration_count = iter + 1;
    BuildFissionSource(phi0_m_, lambda_m, q0_);
    for (size_t i = 0; i < q0_.size(); ++i)
      q0_[i] -= sweep_source_[i];
    source_correction_ = q0_;

    std::fill(operator_phi_.begin(), operator_phi_.end(), 0.0);
    diffusion_acceleration_.AssembleRHS(q0_, boundary_source_, current_correction_);
    diffusion_acceleration_.Solve(operator_phi_, false);

    for (size_t i = 0; i < phi0_kp1_.size(); ++i)
      phi0_kp1_[i] = phi0_star_[i] + operator_phi_[i];
    diffusion_acceleration_.ProjectScalarFlux(phi0_kp1_, phi_temp_);

    const double production_kp1 = ComputeFissionProduction(do_problem_, phi_temp_);
    OpenSnLogicalErrorIf(production_m == 0.0,
                         "UDSA k-eigenvalue acceleration encountered zero low-order production.");
    const double lambda_kp1 = production_kp1 / production_m * lambda_m;

    change = std::fabs(lambda_kp1 - lambda_m) / std::max(std::fabs(lambda_kp1), 1.0e-16);
    phi0_m_.swap(phi0_kp1_);
    production_m = production_kp1;
    lambda_m = lambda_kp1;
    if (change < pi_k_tol_)
      break;
  }

  phi0_ = phi0_m_;

  if (verbose_)
  {
    for (size_t i = 0; i < operator_phi_.size(); ++i)
      operator_phi_[i] = phi0_[i] - phi0_star_[i];

    std::ostringstream out;
    out << "PIUDSA delta solve";
    AppendNumericField(out, "iterations", iteration_count);
    AppendNumericField(out, "change", change, Scientific(6));
    AppendNumericField(out, "transport_k", transport_lambda, Fixed(7));
    AppendNumericField(out, "low_order_k", lambda_m, Fixed(7));
    AppendNumericField(out, "flux_correction_l2", GlobalL2Norm(operator_phi_), Scientific(6));
    AppendNumericField(out, "fission_delta_l2", GlobalL2Norm(source_correction_), Scientific(6));
    log.Log() << program_timer.GetTimeString() << " " << out.str();
  }

  diffusion_acceleration_.ProjectScalarFlux(phi0_, phi_new_local_);
  const double production = ComputeFissionProduction(do_problem_, phi_new_local_);
  const double output_lambda = use_transport_eigenvalue_ ? transport_lambda : lambda_m;
  if (production > 0.0)
    LBSVecOps::ScalePhiVector(do_problem_, PhiSTLOption::PHI_NEW, output_lambda / production);

  LBSVecOps::GSScopedCopyPrimarySTLvectors(
    do_problem_, front_gs_, PhiSTLOption::PHI_NEW, PhiSTLOption::PHI_OLD);

  return output_lambda;
}

} // namespace opensn
