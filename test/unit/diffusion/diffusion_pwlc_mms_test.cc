// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "test/unit/common/mesh_builders.h"
#include "modules/diffusion/diffusion_pwlc_solver.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/acceleration.h"
#include "framework/math/spatial_discretization/finite_element/piecewise_linear/piecewise_linear_continuous.h"
#include "framework/math/spatial_discretization/finite_element/unit_cell_matrices.h"
#include "framework/runtime.h"

#include <cmath>
#include <map>
#include <vector>

using namespace opensn;

namespace
{

double
ExactSolution(const Vector3&)
{
  return 1.0;
}

double
ComputeL2Error(const SpatialDiscretization& sdm,
               const MeshContinuum& grid,
               const std::vector<double>& solution)
{
  double local_error = 0.0;
  for (const auto& cell : grid.GetLocalCells())
  {
    const auto& mapping = sdm.GetCellMapping(*cell);
    const auto fe_data = mapping.MakeVolumetricFiniteElementData();
    const auto num_nodes = mapping.GetNumNodes();

    for (const auto qp : fe_data.GetQuadraturePointIndices())
    {
      double numerical_solution = 0.0;
      for (size_t j = 0; j < num_nodes; ++j)
        numerical_solution += solution[sdm.MapDOFLocal(*cell, j)] * fe_data.ShapeValue(j, qp);

      const double error = numerical_solution - ExactSolution(fe_data.QPointXYZ(qp));
      local_error += error * error * fe_data.JxW(qp);
    }
  }

  double global_error = 0.0;
  mpi_comm.all_reduce(local_error, global_error, mpi::op::sum<double>());
  return std::sqrt(global_error);
}

} // namespace

TEST(DiffusionPWLC, ReproducesConstantManufacturedSolution)
{
  if (mpi_comm.size() != 1)
    GTEST_SKIP() << "PWLC manufactured-solution test is serial only.";

  auto grid = BuildSquareMesh(2.0, 8, -1.0);
  grid->SetUniformBlockID(0);
  grid->SetOrthogonalBoundaries();

  auto sdm = PieceWiseLinearContinuous::New(grid);
  const auto& unknown_manager = sdm->UNITARY_UNKNOWN_MANAGER;

  std::vector<UnitCellMatrices> unit_matrices(grid->GetLocalCellCount());
  for (const auto& cell : grid->GetLocalCells())
    unit_matrices[cell->local_id] = ComputeUnitCellIntegrals(*sdm, *cell);

  // For phi=1, D=1, and Sigma_r=1, -div(D grad(phi))+Sigma_r phi=1.
  // The exact normal current is zero on every boundary.
  std::map<uint64_t, BoundaryCondition> boundary_conditions = {
    {XMIN, {BCType::ROBIN, {0.0, 1.0, 0.0}}},
    {XMAX, {BCType::ROBIN, {0.0, 1.0, 0.0}}},
    {YMIN, {BCType::ROBIN, {0.0, 1.0, 0.0}}},
    {YMAX, {BCType::ROBIN, {0.0, 1.0, 0.0}}},
  };
  std::map<unsigned int, Multigroup_D_and_sigR> material = {
    {0, Multigroup_D_and_sigR{{1.0}, {1.0}}}};

  DiffusionPWLCSolver solver("PWLC affine MMS",
                             *sdm,
                             unknown_manager,
                             boundary_conditions,
                             material,
                             unit_matrices,
                             false,
                             false);
  solver.options.residual_tolerance = 1.0e-12;
  solver.options.perform_symmetry_check = true;
  solver.Initialize();

  std::vector<double> source(sdm->GetNumLocalAndGhostDOFs(unknown_manager), 0.0);
  for (const auto& cell : grid->GetLocalCells())
  {
    const auto& mapping = sdm->GetCellMapping(*cell);
    const auto& nodes = mapping.GetNodeLocations();
    for (size_t j = 0; j < nodes.size(); ++j)
      source[sdm->MapDOFLocal(*cell, j)] = ExactSolution(nodes[j]);
  }

  std::vector<double> solution(source.size(), 0.0);
  solver.AssembleAand_b(source);
  solver.Solve(solution);
  EXPECT_LT(ComputeL2Error(*sdm, *grid, solution), 1.0e-10);

  std::fill(solution.begin(), solution.end(), 0.0);
  solver.Assemble_b(source);
  solver.Solve(solution);
  EXPECT_LT(ComputeL2Error(*sdm, *grid, solution), 1.0e-10);
}
