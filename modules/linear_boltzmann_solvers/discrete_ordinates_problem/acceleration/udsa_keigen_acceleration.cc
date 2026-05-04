// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/udsa_keigen_acceleration.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/solvers/pi_keigen_solver.h"
#include "modules/diffusion/diffusion_mip_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/compute/lbs_compute.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/iteration_logging.h"
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

void
CopyLocalPetscVector(const Vec& petsc_vector, std::vector<double>& values)
{
  PetscInt local_size = 0;
  OpenSnPETScCall(VecGetLocalSize(petsc_vector, &local_size));
  values.assign(static_cast<size_t>(local_size), 0.0);

  const PetscScalar* data = nullptr;
  OpenSnPETScCall(VecGetArrayRead(petsc_vector, &data));
  for (PetscInt i = 0; i < local_size; ++i)
    values[static_cast<size_t>(i)] = data[i];
  OpenSnPETScCall(VecRestoreArrayRead(petsc_vector, &data));
}

} // namespace

OpenSnRegisterObjectInNamespace(lbs, UDSAKEigenAcceleration);

InputParameters
UDSAKEigenAcceleration::GetInputParameters()
{
  auto params = DiscreteOrdinatesKEigenAcceleration::GetInputParameters();
  params.ChangeExistingParamToOptional("name", "UDSAKEigenAcceleration");
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
    diffusion_acceleration_(do_problem_)
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
  diffusion_acceleration_.SetSolverOptions(l_abs_tol_, max_iters_, verbose_, petsc_options_);
  AddScatterCouplingToOperator();

  const auto num_local_dofs = diffusion_acceleration_.GetNumLocalDOFs();
  phi0_.assign(num_local_dofs, 0.0);
  phi0_ell_.assign(num_local_dofs, 0.0);
  phi0_star_.assign(num_local_dofs, 0.0);
  phi0_m_.assign(num_local_dofs, 0.0);
  phi0_kp1_.assign(num_local_dofs, 0.0);
  q0_.assign(num_local_dofs, 0.0);
  boundary_source_.assign(num_local_dofs, 0.0);
  current_correction_.assign(num_local_dofs, 0.0);
  sweep_source_.assign(num_local_dofs, 0.0);
  sweep_rhs_.assign(num_local_dofs, 0.0);
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
  production_ell_ = ComputeFissionProduction(do_problem_, phi_old_local_);
  diffusion_acceleration_.CopyScalarFlux(phi_old_local_, phi0_ell_);
}

void
UDSAKEigenAcceleration::AddScatterCouplingToOperator()
{
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& udsa_uk_man = diffusion_acceleration_.GetUnknownManager();
  const auto& block_id_to_xs = do_problem_.GetBlockID2XSMap();
  const auto& unit_cell_matrices = do_problem_.GetUnitCellMatrices();
  const auto num_groups = do_problem_.GetNumGroups();

  std::vector<PetscInt> rows;
  std::vector<PetscInt> cols;
  std::vector<PetscScalar> vals;

  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    const auto& xs = *block_id_to_xs.at(cell.block_id);
    const auto* S0 = xs.GetTransferMatrices().empty() ? nullptr : &xs.GetTransferMatrix(0);
    if (S0 == nullptr)
      continue;

    for (size_t i = 0; i < num_nodes; ++i)
    {
      for (unsigned int g = 0; g < num_groups; ++g)
      {
        const auto row = sdm.MapDOF(cell, i, udsa_uk_man, 0, g);

        for (const auto& entry : S0->Row(g))
        {
          const auto gp = static_cast<unsigned int>(entry.column_index);
          if (gp == g)
            continue;

          for (size_t j = 0; j < num_nodes; ++j)
          {
            const auto col = sdm.MapDOF(cell, j, udsa_uk_man, 0, gp);
            const auto& mass = unit_cell_matrices[cell.local_id].intV_shapeI_shapeJ;
            rows.push_back(static_cast<PetscInt>(row));
            cols.push_back(static_cast<PetscInt>(col));
            vals.push_back(static_cast<PetscScalar>(-entry.value * mass(i, j)));
          }
        }
      }
    }
  }

  if (not rows.empty())
    diffusion_acceleration_.diffusion_solver_->AddToMatrix(rows, cols, vals);
}

void
UDSAKEigenAcceleration::BuildFissionSource(const std::vector<double>& phi0,
                                           const double lambda,
                                           std::vector<double>& q0) const
{
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& udsa_uk_man = diffusion_acceleration_.GetUnknownManager();
  const auto& block_id_to_xs = do_problem_.GetBlockID2XSMap();
  const auto num_groups = do_problem_.GetNumGroups();

  q0.assign(sdm.GetNumLocalDOFs(udsa_uk_man), 0.0);
  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    const auto& xs = *block_id_to_xs.at(cell.block_id);
    if (not xs.IsFissionable())
      continue;

    const auto& F = xs.GetProductionMatrix();
    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto udsa_map = sdm.MapDOFLocal(cell, i, udsa_uk_man, 0, 0);

      for (unsigned int g = 0; g < num_groups; ++g)
      {
        double source = 0.0;
        for (unsigned int gp = 0; gp < num_groups; ++gp)
          source += F[g][gp] * phi0[udsa_map + gp] / lambda;

        q0[udsa_map + g] = source;
      }
    }
  }
}

double
UDSAKEigenAcceleration::PostPowerIteration()
{
  CALI_CXX_MARK_SCOPE("UDSAKEigenAcceleration::PostPowerIteration");

  diffusion_acceleration_.CopyScalarFlux(phi_new_local_, phi0_star_);
  const double lambda = solver_->GetEigenvalue();
  const double transport_production = ComputeFissionProduction(do_problem_, phi_new_local_);
  OpenSnLogicalErrorIf(production_ell_ == 0.0,
                       "UDSA k-eigenvalue acceleration encountered zero previous production.");
  const double transport_lambda = transport_production / production_ell_ * lambda;

  diffusion_acceleration_.BuildBoundarySource(boundary_source_);
  std::fill(current_correction_.begin(), current_correction_.end(), 0.0);
  BuildFissionSource(phi0_ell_, lambda, sweep_source_);
  diffusion_acceleration_.AssembleRHS(sweep_source_, boundary_source_, current_correction_);
  CopyLocalPetscVector(diffusion_acceleration_.diffusion_solver_->GetRHS(), sweep_rhs_);

  diffusion_acceleration_.diffusion_solver_->ApplyOperator(phi0_star_, operator_phi_);
  for (size_t i = 0; i < current_correction_.size(); ++i)
    current_correction_[i] = operator_phi_[i] - sweep_rhs_[i];

  phi0_m_ = phi0_star_;
  double production_m = transport_production;
  double lambda_m = transport_lambda;

  double change = 0.0;
  unsigned int iter = 0;
  for (; iter < static_cast<unsigned int>(std::max(pi_max_its_, 1)); ++iter)
  {
    BuildFissionSource(phi0_m_, lambda_m, q0_);
    diffusion_acceleration_.AssembleRHS(q0_, boundary_source_, current_correction_);

    phi0_kp1_ = phi0_m_;
    diffusion_acceleration_.Solve(phi0_kp1_, false);
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
    std::ostringstream out;
    out << "PIUDSA split solve";
    AppendNumericField(out, "iterations", iter + 1);
    AppendNumericField(out, "change", change, Scientific(6));
    AppendNumericField(out, "correction_l2", GlobalL2Norm(current_correction_), Scientific(6));
    log.Log() << program_timer.GetTimeString() << " " << out.str();
  }

  diffusion_acceleration_.ProjectScalarFlux(phi0_, phi_new_local_);
  const double production = ComputeFissionProduction(do_problem_, phi_new_local_);
  if (production > 0.0)
    LBSVecOps::ScalePhiVector(do_problem_, PhiSTLOption::PHI_NEW, lambda_m / production);

  LBSVecOps::GSScopedCopyPrimarySTLvectors(
    do_problem_, front_gs_, PhiSTLOption::PHI_NEW, PhiSTLOption::PHI_OLD);

  return lambda_m;
}

} // namespace opensn
