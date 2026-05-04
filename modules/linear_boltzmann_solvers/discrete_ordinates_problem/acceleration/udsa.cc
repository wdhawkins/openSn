// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/udsa.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/acceleration.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_residual.h"
#include "modules/diffusion/diffusion_mip_solver.h"
#include "framework/data_types/sparse_matrix/sparse_matrix.h"
#include "framework/math/quadratures/angular/angular_quadrature.h"
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

std::map<uint64_t, BoundaryCondition>
MakeUDSABoundaryConditions(
  const std::map<uint64_t, std::shared_ptr<SweepBoundary>>& sweep_boundaries)
{
  std::map<uint64_t, BoundaryCondition> bcs;
  for (const auto& [bid, boundary] : sweep_boundaries)
  {
    if (boundary->GetType() == LBSBoundaryType::REFLECTING)
      bcs[bid] = {BCType::ROBIN, {0.0, 1.0, 0.0}};
    else
      bcs[bid] = {BCType::ROBIN, {0.25, 0.5, 0.0}};
  }
  return bcs;
}

} // namespace

UDSADiffusionAcceleration::UDSADiffusionAcceleration(DiscreteOrdinatesProblem& do_problem,
                                                     const ScatterCouplingMode scatter_coupling_mode)
  : do_problem_(do_problem), scatter_coupling_mode_(scatter_coupling_mode)
{
}

void
UDSADiffusionAcceleration::Initialize()
{
  CALI_CXX_MARK_SCOPE("UDSADiffusionAcceleration::Initialize");

  const auto& options = do_problem_.GetOptions();

  UnknownManager uk_man;
  uk_man.AddUnknown(UnknownType::VECTOR_N, do_problem_.GetNumGroups());
  if (scatter_coupling_mode_ == ScatterCouplingMode::Operator)
    uk_man.SetUnknownNumOffBlockConnections(0, do_problem_.GetNumGroups() - 1);

  const auto& sweep_boundaries = do_problem_.GetSweepBoundaries();
  auto bcs = MakeUDSABoundaryConditions(sweep_boundaries);

  auto matid_2_mgxs_map =
    PackGroupsetXS(do_problem_.GetBlockID2XSMap(), 0, do_problem_.GetNumGroups() - 1);

  const auto& sdm = do_problem_.GetSpatialDiscretization();
  diffusion_solver_ = std::make_shared<DiffusionMIPSolver>(do_problem_.GetName() + "_UDSA",
                                                           sdm,
                                                           uk_man,
                                                           bcs,
                                                           matid_2_mgxs_map,
                                                           do_problem_.GetUnitCellMatrices(),
                                                           false,
                                                           options.udsa_verbose);

  diffusion_solver_->options.residual_tolerance = options.udsa_tol;
  diffusion_solver_->options.max_iters = options.udsa_max_iters;
  diffusion_solver_->options.verbose = options.udsa_verbose;
  diffusion_solver_->options.additional_options_string = options.udsa_string;

  diffusion_solver_->Initialize();

  sweep_residual_evaluator_ = std::make_shared<SweepResidualEvaluator>(do_problem_);

  std::vector<double> dummy_rhs(sdm.GetNumLocalDOFs(uk_man), 0.0);
  diffusion_solver_->AssembleAand_b(dummy_rhs);
  if (scatter_coupling_mode_ == ScatterCouplingMode::Operator)
    AddScatterCouplingToOperator();
}

const UnknownManager&
UDSADiffusionAcceleration::GetUnknownManager() const
{
  OpenSnLogicalErrorIf(not diffusion_solver_,
                       "UDSADiffusionAcceleration: requested unknown manager before Initialize.");
  return diffusion_solver_->GetUnknownStructure();
}

void
UDSADiffusionAcceleration::SetSolverOptions(const double residual_tolerance,
                                            const unsigned int max_iters,
                                            const bool verbose,
                                            const std::string& petsc_options)
{
  OpenSnLogicalErrorIf(not diffusion_solver_,
                       "UDSADiffusionAcceleration: requested solver options before Initialize.");
  diffusion_solver_->options.residual_tolerance = residual_tolerance;
  diffusion_solver_->options.max_iters = max_iters;
  diffusion_solver_->options.verbose = verbose;
  diffusion_solver_->options.additional_options_string = petsc_options;
  diffusion_solver_->ApplyOptions();
}

void
UDSADiffusionAcceleration::CopyScalarFlux(const std::vector<double>& phi_in,
                                          std::vector<double>& phi0) const
{
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& phi_uk_man = do_problem_.GetUnknownManager();
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();
  const auto num_groups = do_problem_.GetNumGroups();

  phi0.assign(sdm.GetNumLocalDOFs(udsa_uk_man), 0.0);
  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);
      const auto udsa_map = sdm.MapDOFLocal(cell, i, udsa_uk_man, 0, 0);
      for (unsigned int g = 0; g < num_groups; ++g)
        phi0[udsa_map + g] = phi_in[phi_map + g];
    }
  }
}

void
UDSADiffusionAcceleration::ProjectScalarFlux(const std::vector<double>& phi0,
                                             std::vector<double>& phi_out) const
{
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& phi_uk_man = do_problem_.GetUnknownManager();
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();
  const auto num_groups = do_problem_.GetNumGroups();

  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);
      const auto udsa_map = sdm.MapDOFLocal(cell, i, udsa_uk_man, 0, 0);
      for (unsigned int g = 0; g < num_groups; ++g)
        phi_out[phi_map + g] = phi0[udsa_map + g];
    }
  }
}

void
UDSADiffusionAcceleration::BuildFixedSourceMoments(std::vector<double>& source_moments) const
{
  source_moments.assign(do_problem_.GetPhiNewLocal().size(), 0.0);
  const std::vector<double> zero_phi(do_problem_.GetPhiNewLocal().size(), 0.0);
  const auto set_source_function = do_problem_.GetActiveSetSourceFunction();
  for (const auto& groupset : do_problem_.GetGroupsets())
    set_source_function(groupset, source_moments, zero_phi, APPLY_FIXED_SOURCES);
}

void
UDSADiffusionAcceleration::AddScatterCouplingToOperator()
{
  // The k-eigenvalue low-order equation uses (A - S_off) on the left-hand side.
  // Fixed-source UDSA keeps this coupling on the RHS instead.
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();
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

    const auto& mass = unit_cell_matrices[cell.local_id].intV_shapeI_shapeJ;
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
            rows.push_back(static_cast<PetscInt>(row));
            cols.push_back(static_cast<PetscInt>(col));
            vals.push_back(static_cast<PetscScalar>(-entry.value * mass(i, j)));
          }
        }
      }
    }
  }

  diffusion_solver_->AddToMatrix(rows, cols, vals);
}

void
UDSADiffusionAcceleration::BuildDiffusionSource(const std::vector<double>& phi0,
                                                const std::vector<double>& fixed_source_moments,
                                                std::vector<double>& q0) const
{
  OpenSnLogicalErrorIf(scatter_coupling_mode_ != ScatterCouplingMode::RHS,
                       "BuildDiffusionSource is only valid when UDSA scatter coupling is on the "
                       "right-hand side.");

  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& phi_uk_man = do_problem_.GetUnknownManager();
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();
  const auto& block_id_to_xs = do_problem_.GetBlockID2XSMap();
  const auto num_groups = do_problem_.GetNumGroups();

  q0.assign(sdm.GetNumLocalDOFs(udsa_uk_man), 0.0);
  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    const auto& xs = *block_id_to_xs.at(cell.block_id);
    const auto* S0 = xs.GetTransferMatrices().empty() ? nullptr : &xs.GetTransferMatrix(0);
    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto phi_map = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);
      const auto udsa_map = sdm.MapDOFLocal(cell, i, udsa_uk_man, 0, 0);

      for (unsigned int g = 0; g < num_groups; ++g)
      {
        q0[udsa_map + g] = fixed_source_moments[phi_map + g];
        if (S0 == nullptr)
          continue;

        for (const auto& entry : S0->Row(g))
        {
          const auto gp = static_cast<unsigned int>(entry.column_index);
          if (gp == g)
            continue;

          q0[udsa_map + g] += entry.value * phi0[udsa_map + gp];
        }
      }
    }
  }
}

void
UDSADiffusionAcceleration::AddScatterCouplingSource(const std::vector<double>& phi0,
                                                    std::vector<double>& source) const
{
  // BuildCurrentCorrection applies the runtime operator. In Operator mode that
  // operator contains -S_off, so add S_off phi here to recover the unsplit
  // diffusion/removal contribution used by the transport residual correction.
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();
  const auto& block_id_to_xs = do_problem_.GetBlockID2XSMap();
  const auto& unit_cell_matrices = do_problem_.GetUnitCellMatrices();
  const auto num_groups = do_problem_.GetNumGroups();

  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    const auto& xs = *block_id_to_xs.at(cell.block_id);
    const auto* S0 = xs.GetTransferMatrices().empty() ? nullptr : &xs.GetTransferMatrix(0);
    if (S0 == nullptr)
      continue;

    const auto& mass = unit_cell_matrices[cell.local_id].intV_shapeI_shapeJ;
    for (size_t i = 0; i < num_nodes; ++i)
    {
      for (unsigned int g = 0; g < num_groups; ++g)
      {
        const auto row = sdm.MapDOFLocal(cell, i, udsa_uk_man, 0, g);

        for (const auto& entry : S0->Row(g))
        {
          const auto gp = static_cast<unsigned int>(entry.column_index);
          if (gp == g)
            continue;

          for (size_t j = 0; j < num_nodes; ++j)
          {
            const auto col = sdm.MapDOFLocal(cell, j, udsa_uk_man, 0, gp);
            source[row] += entry.value * mass(i, j) * phi0[col];
          }
        }
      }
    }
  }
}

void
UDSADiffusionAcceleration::BuildBoundarySource(std::vector<double>& boundary_source) const
{
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();
  const auto& sweep_boundaries = do_problem_.GetSweepBoundaries();

  boundary_source.assign(sdm.GetNumLocalDOFs(udsa_uk_man), 0.0);
  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto& unit_cell_matrices = do_problem_.GetUnitCellMatrices()[cell.local_id];

    for (size_t f = 0; f < cell.faces.size(); ++f)
    {
      const auto& face = cell.faces[f];
      if (face.has_neighbor)
        continue;

      const auto boundary_it = sweep_boundaries.find(face.neighbor_id);
      if (boundary_it == sweep_boundaries.end())
        continue;

      auto& boundary = *boundary_it->second;
      if (boundary.GetType() == LBSBoundaryType::VACUUM or
          boundary.GetType() == LBSBoundaryType::REFLECTING)
        continue;

      const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
      const auto& intS_shapeI = unit_cell_matrices.intS_shapeI[f];
      for (const auto& groupset : do_problem_.GetGroupsets())
      {
        const auto& quadrature = *groupset.quadrature;
        for (unsigned int g = groupset.first_group; g <= groupset.last_group; ++g)
        {
          for (size_t fi = 0; fi < num_face_nodes; ++fi)
          {
            const int i = cell_mapping.MapFaceNode(f, fi);
            double incoming_partial_current = 0.0;
            for (size_t n = 0; n < quadrature.omegas.size(); ++n)
            {
              const double mu = quadrature.omegas[n].Dot(face.normal);
              if (mu >= 0.0)
                continue;

              const double psi = *boundary.PsiIncoming(cell.local_id,
                                                       static_cast<unsigned int>(f),
                                                       static_cast<unsigned int>(fi),
                                                       static_cast<unsigned int>(n),
                                                       g);
              incoming_partial_current += -mu * quadrature.weights[n] * psi;
            }

            const auto udsa_i = sdm.MapDOFLocal(cell, i, udsa_uk_man, 0, g);
            boundary_source[udsa_i] += 2.0 * incoming_partial_current * intS_shapeI(i);
          }
        }
      }
    }
  }
}

void
UDSADiffusionAcceleration::BuildCurrentCorrection(const std::vector<double>& phi0,
                                                  const std::vector<double>& boundary_source,
                                                  std::vector<double>& correction)
{
  const auto& udsa_uk_man = diffusion_solver_->GetUnknownStructure();

  OpenSnLogicalErrorIf(not do_problem_.GetOptions().save_angular_flux,
                       do_problem_.GetName() + ": UDSA requires options.save_angular_flux=true.");

  diffusion_solver_->ApplyOperator(phi0, correction);
  if (scatter_coupling_mode_ == ScatterCouplingMode::Operator)
    AddScatterCouplingSource(phi0, correction);
  for (size_t i = 0; i < correction.size(); ++i)
    correction[i] -= boundary_source[i];

  std::vector<double> removal;
  std::vector<double> volume_streaming;
  std::vector<double> face_streaming;
  sweep_residual_evaluator_->AddStreamingResidualComponents(
    udsa_uk_man, phi0, removal, volume_streaming, face_streaming);
  for (size_t i = 0; i < correction.size(); ++i)
    correction[i] += removal[i] + volume_streaming[i] + face_streaming[i];
}

void
UDSADiffusionAcceleration::AssembleRHS(const std::vector<double>& q0,
                                       const std::vector<double>& boundary_source,
                                       const std::vector<double>& current_correction)
{
  diffusion_solver_->Assemble_b(q0);
  diffusion_solver_->AddToRHS(boundary_source);
  diffusion_solver_->AddToRHS(current_correction);
}

void
UDSADiffusionAcceleration::Solve(std::vector<double>& phi0, const bool use_initial_guess)
{
  diffusion_solver_->Solve(phi0, use_initial_guess);
}

double
UDSADiffusionAcceleration::ComputeL2Change(const std::vector<double>& current,
                                           const std::vector<double>& previous) const
{
  double local_delta = 0.0;
  double local_ref = 0.0;
  for (size_t i = 0; i < current.size(); ++i)
  {
    const double delta = current[i] - previous[i];
    local_delta += delta * delta;
    local_ref += current[i] * current[i];
  }

  double global_delta = 0.0;
  double global_ref = 0.0;
  opensn::mpi_comm.all_reduce(&local_delta, 1, &global_delta, mpi::op::sum<double>());
  opensn::mpi_comm.all_reduce(&local_ref, 1, &global_ref, mpi::op::sum<double>());

  return std::sqrt(global_delta) / std::max(std::sqrt(global_ref), 1.0e-16);
}

size_t
UDSADiffusionAcceleration::GetNumLocalDOFs() const
{
  return do_problem_.GetSpatialDiscretization().GetNumLocalDOFs(GetUnknownManager());
}

UDSAAcceleration::UDSAAcceleration(DiscreteOrdinatesProblem& do_problem)
  : do_problem_(do_problem), diffusion_acceleration_(do_problem)
{
}

void
UDSAAcceleration::Initialize()
{
  CALI_CXX_MARK_SCOPE("UDSAAcceleration::Initialize");

  diffusion_acceleration_.Initialize();

  if (not diffusion_acceleration_.IsInitialized())
    return;

  const auto num_local_dofs = diffusion_acceleration_.GetNumLocalDOFs();
  q0_.assign(num_local_dofs, 0.0);
  phi0_.assign(num_local_dofs, 0.0);
  phi0_old_.assign(num_local_dofs, 0.0);
  fixed_source_moments_.assign(do_problem_.GetPhiNewLocal().size(), 0.0);
  boundary_source_.assign(num_local_dofs, 0.0);
  current_correction_.assign(num_local_dofs, 0.0);
}

void
UDSAAcceleration::Apply()
{
  CALI_CXX_MARK_SCOPE("UDSAAcceleration::Apply");

  if (not diffusion_acceleration_.IsInitialized())
    return;

  const auto& options = do_problem_.GetOptions();
  if (options.udsa_max_iters == 0)
    return;

  diffusion_acceleration_.CopyScalarFlux(do_problem_.GetPhiNewLocal(), phi0_);
  diffusion_acceleration_.BuildFixedSourceMoments(fixed_source_moments_);
  diffusion_acceleration_.BuildBoundarySource(boundary_source_);
  diffusion_acceleration_.BuildCurrentCorrection(phi0_, boundary_source_, current_correction_);

  double change = 0.0;
  unsigned int iter = 0;
  for (; iter < options.udsa_max_iters; ++iter)
  {
    phi0_old_ = phi0_;
    diffusion_acceleration_.BuildDiffusionSource(phi0_old_, fixed_source_moments_, q0_);
    diffusion_acceleration_.AssembleRHS(q0_, boundary_source_, current_correction_);
    diffusion_acceleration_.Solve(phi0_, iter > 0);

    change = diffusion_acceleration_.ComputeL2Change(phi0_, phi0_old_);
    if (change < options.udsa_tol)
      break;
  }

  if (options.udsa_verbose)
  {
    std::stringstream out;
    out << program_timer.GetTimeString() << " UDSA final";
    out << ", iterations = " << (iter + 1);
    out << ", l2_change = " << std::scientific << change;
    log.Log() << out.str();
  }

  auto& phi_new = do_problem_.GetPhiNewLocal();
  auto& phi_old = do_problem_.GetPhiOldLocal();
  diffusion_acceleration_.ProjectScalarFlux(phi0_, phi_new);
  diffusion_acceleration_.ProjectScalarFlux(phi0_, phi_old);
}

} // namespace opensn
