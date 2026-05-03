// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_residual.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include "framework/data_types/sparse_matrix/sparse_matrix.h"
#include "framework/data_types/vector_ghost_communicator/vector_ghost_communicator.h"
#include "framework/math/quadratures/angular/angular_quadrature.h"
#include "framework/runtime.h"
#include <set>

namespace opensn
{

namespace
{

std::vector<uint64_t>
MakePWLDGhostIndices(const SpatialDiscretization& sdm, const UnknownManager& uk_man)
{
  std::set<uint64_t> ghost_ids;
  const auto& grid = *sdm.GetGrid();
  for (const uint64_t ghost_id : grid.cells.GetGhostGlobalIDs())
  {
    const auto& cell = grid.cells[ghost_id];
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    for (int i = 0; i < cell_mapping.GetNumNodes(); ++i)
      for (int u = 0; u < uk_man.GetNumberOfUnknowns(); ++u)
        for (int c = 0; c < uk_man.unknowns[u].GetNumComponents(); ++c)
          ghost_ids.insert(sdm.MapDOF(cell, i, uk_man, u, c));
  }
  return {ghost_ids.begin(), ghost_ids.end()};
}

int
MapAssociatedFaceNode(const Vector3& node,
                      const std::vector<Vector3>& neighbor_nodes,
                      const double epsilon = 1.0e-12)
{
  for (int j = 0; j < neighbor_nodes.size(); ++j)
    if ((node - neighbor_nodes[j]).NormSquare() < epsilon)
      return j;

  throw std::logic_error("SweepResidualEvaluator: Associated neighbor node not found.");
}

} // namespace

SweepResidualEvaluator::SweepResidualEvaluator(DiscreteOrdinatesProblem& do_problem)
  : do_problem_(do_problem)
{
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  psi_ghost_communicators_.resize(do_problem_.GetGroupsets().size());
  ghosted_psi_new_local_.resize(do_problem_.GetGroupsets().size());

  for (const auto& groupset : do_problem_.GetGroupsets())
  {
    const auto local_size = sdm.GetNumLocalDOFs(groupset.psi_uk_man_);
    const auto global_size = sdm.GetNumGlobalDOFs(groupset.psi_uk_man_);
    const auto ghost_ids = MakePWLDGhostIndices(sdm, groupset.psi_uk_man_);
    psi_ghost_communicators_[groupset.id] = std::make_shared<VectorGhostCommunicator>(
      local_size, global_size, ghost_ids, opensn::mpi_comm);
  }
}

void
SweepResidualEvaluator::AddStreamingResidual(const UnknownManager& residual_uk_man,
                                             const std::vector<double>& phi0,
                                             std::vector<double>& residual)
{
  const auto grid = do_problem_.GetGrid();
  const auto& sdm = do_problem_.GetSpatialDiscretization();
  const auto& block_id_to_xs = do_problem_.GetBlockID2XSMap();
  const auto& psi_new_local = do_problem_.GetPsiNewLocal();
  const auto& sweep_boundaries = do_problem_.GetSweepBoundaries();

  OpenSnLogicalErrorIf(not do_problem_.GetOptions().save_angular_flux,
                       do_problem_.GetName() +
                         ": sweep residual evaluation requires saved angular fluxes.");

  for (const auto& groupset : do_problem_.GetGroupsets())
  {
    auto& ghosted_psi = ghosted_psi_new_local_[groupset.id];
    ghosted_psi =
      psi_ghost_communicators_[groupset.id]->MakeGhostedVector(psi_new_local.at(groupset.id));
    psi_ghost_communicators_[groupset.id]->CommunicateGhostEntries(ghosted_psi);
  }

  for (const auto& cell : grid->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const auto num_nodes = cell_mapping.GetNumNodes();
    const auto nodes = cell_mapping.GetNodeLocations();
    const auto& unit_cell_matrices = do_problem_.GetUnitCellMatrices()[cell.local_id];
    const auto& intV_shapeI_shapeJ = unit_cell_matrices.intV_shapeI_shapeJ;
    const auto& intV_shapeI_gradshapeJ = unit_cell_matrices.intV_shapeI_gradshapeJ;
    const auto& intS_shapeI_shapeJ = unit_cell_matrices.intS_shapeI_shapeJ;
    const auto& sigma_removal = block_id_to_xs.at(cell.block_id)->GetSigmaRemoval();

    for (const auto& groupset : do_problem_.GetGroupsets())
    {
      const auto& quadrature = *groupset.quadrature;
      const auto& psi = ghosted_psi_new_local_[groupset.id];
      const auto& psi_uk_man = groupset.psi_uk_man_;
      const auto& psi_ghost_comm = *psi_ghost_communicators_[groupset.id];

      for (unsigned int g = groupset.first_group; g <= groupset.last_group; ++g)
      {
        const auto gsg = g - groupset.first_group;
        for (size_t i = 0; i < num_nodes; ++i)
        {
          const auto residual_i = sdm.MapDOFLocal(cell, i, residual_uk_man, 0, g);
          double rhs_i = 0.0;

          for (size_t j = 0; j < num_nodes; ++j)
          {
            const auto residual_j = sdm.MapDOFLocal(cell, j, residual_uk_man, 0, g);
            rhs_i -= sigma_removal[g] * intV_shapeI_shapeJ(i, j) * phi0[residual_j];
            for (size_t n = 0; n < quadrature.omegas.size(); ++n)
            {
              const auto psi_j = sdm.MapDOFLocal(cell, j, psi_uk_man, n, 0);
              const double streaming = quadrature.weights[n] *
                                       intV_shapeI_gradshapeJ(i, j).Dot(quadrature.omegas[n]) *
                                       psi[psi_j + gsg];
              rhs_i -= streaming;
            }
          }

          residual[residual_i] += rhs_i;
        }

        for (size_t f = 0; f < cell.faces.size(); ++f)
        {
          const auto& face = cell.faces[f];
          const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
          const Cell* neighbor_cell = nullptr;
          std::vector<Vector3> neighbor_nodes;
          if (face.has_neighbor)
          {
            neighbor_cell = &grid->cells[face.neighbor_id];
            neighbor_nodes = sdm.GetCellMapping(*neighbor_cell).GetNodeLocations();
          }

          for (size_t fi = 0; fi < num_face_nodes; ++fi)
          {
            const int i = cell_mapping.MapFaceNode(f, fi);
            const auto residual_i = sdm.MapDOFLocal(cell, i, residual_uk_man, 0, g);
            double rhs_i = 0.0;

            for (size_t fj = 0; fj < num_face_nodes; ++fj)
            {
              const int j = cell_mapping.MapFaceNode(f, fj);

              for (size_t n = 0; n < quadrature.omegas.size(); ++n)
              {
                const double mu = quadrature.omegas[n].Dot(face.normal);
                if (mu >= 0.0)
                  continue;

                const auto psi_cell_j = sdm.MapDOFLocal(cell, j, psi_uk_man, n, 0);
                double psi_upwind = 0.0;

                if (face.has_neighbor)
                {
                  const int jp = MapAssociatedFaceNode(nodes[j], neighbor_nodes);
                  const auto psi_neighbor_j =
                    grid->IsCellLocal(face.neighbor_id)
                      ? sdm.MapDOFLocal(*neighbor_cell, jp, psi_uk_man, n, 0)
                      : psi_ghost_comm
                          .MapGhostToLocal(sdm.MapDOF(*neighbor_cell, jp, psi_uk_man, n, 0))
                          .value();
                  psi_upwind = psi[psi_neighbor_j + gsg];
                }
                else
                {
                  const auto boundary_it = sweep_boundaries.find(face.neighbor_id);
                  if (boundary_it != sweep_boundaries.end() and
                      boundary_it->second->GetType() != LBSBoundaryType::VACUUM)
                  {
                    psi_upwind = *boundary_it->second->PsiIncoming(cell.local_id,
                                                                   static_cast<unsigned int>(f),
                                                                   static_cast<unsigned int>(fj),
                                                                   static_cast<unsigned int>(n),
                                                                   g);
                  }
                }

                const double jump_streaming = quadrature.weights[n] * (-mu) *
                                              intS_shapeI_shapeJ[f](i, j) *
                                              (psi[psi_cell_j + gsg] - psi_upwind);
                rhs_i -= jump_streaming;
              }
            }

            residual[residual_i] += rhs_i;
          }
        }
      }
    }
  }
}

} // namespace opensn
