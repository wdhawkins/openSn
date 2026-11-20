// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "test/unit/common/mesh_builders.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/compute/lbs_compute.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"
#include "framework/math/quadratures/angular/product_quadrature.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/logical_volume/rpp_logical_volume.h"
#include "framework/parameters/parameter_block.h"
#include "framework/runtime.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace opensn;

namespace
{

std::pair<std::shared_ptr<DiscreteOrdinatesProblem>, std::shared_ptr<MultiGroupXS>>
MakeProductionProblem(const bool adjoint, const bool delayed)
{
  auto grid = BuildSquareMesh(1.0, 4, 0.0);
  grid->SetUniformBlockID(0);
  ParameterBlock right_half;
  right_half.AddParameter("xmin", 0.5);
  right_half.AddParameter("xmax", 1.0);
  right_half.AddParameter("infy", true);
  right_half.AddParameter("infz", true);
  grid->SetBlockIDFromLogicalVolume(*RPPLogicalVolume::Create(right_half), 1, true);

  ParameterBlock groupset("0");
  groupset.AddParameter(ParameterBlock("groups_from_to", std::vector<int>{0, 1}));
  groupset.AddParameter(
    "angular_quadrature",
    std::shared_ptr<AngularQuadrature>(std::make_shared<GLCProductQuadrature2DXY>(2, 4, 0)));
  ParameterBlock groupsets("groupsets");
  groupsets.AddParameter(groupset);
  groupsets.ChangeToArray();

  ParameterBlock xs_map("xs_map");
  std::shared_ptr<MultiGroupXS> right_xs;
  for (int block = 0; block < 2; ++block)
  {
    const std::string filename =
      block == 0 ? "xs2g_delayed_crit_1p.cxs" : "xs2g_delayed_density_up_1p.cxs";
    auto xs = std::make_shared<MultiGroupXS>(
      MultiGroupXS::LoadFromOpenSn(std::string(OPENSN_TEST_ROOT) + "/assets/xs/" + filename));
    if (block == 1)
      right_xs = xs;
    ParameterBlock entry(std::to_string(block));
    entry.AddParameter(ParameterBlock("block_ids", std::vector<int>{block}));
    entry.AddParameter("xs", xs);
    xs_map.AddParameter(entry);
  }
  xs_map.ChangeToArray();
  ParameterBlock options("options");
  options.AddParameter("adjoint", adjoint);
  options.AddParameter("use_precursors", delayed);
  ParameterBlock params;
  params.AddParameter("mesh", grid);
  params.AddParameter("num_groups", 2);
  params.AddParameter(groupsets);
  params.AddParameter(xs_map);
  params.AddParameter(options);
  return {DiscreteOrdinatesProblem::Create(params), right_xs};
}

} // namespace

TEST(FissionProductionTest, AnalyticAffineFluxWithDelayedAdjointAndScaledXS)
{
  for (const bool adjoint : {false, true})
    for (const bool delayed : {false, true})
    {
      SCOPED_TRACE(::testing::Message() << "adjoint=" << adjoint << ", delayed=" << delayed);
      auto [problem, right_xs] = MakeProductionProblem(adjoint, delayed);
      const auto& sdm = problem->GetSpatialDiscretization();
      const auto& unknowns = problem->GetUnknownManager();
      auto phi = problem->GetPhiNewLocal();
      for (const auto& cell : problem->GetGrid()->GetLocalCells())
      {
        const auto& nodes = sdm.GetCellNodeLocations(*cell);
        for (unsigned int i = 0; i < nodes.size(); ++i)
        {
          const auto& p = nodes[i];
          phi[sdm.MapDOFLocal(*cell, i, unknowns, 0, 0)] = 1.0 + p.x + 2.0 * p.y;
          // Signed scalar nodal values must be integrated, not clipped or discarded.
          phi[sdm.MapDOFLocal(*cell, i, unknowns, 0, 1)] = -0.5 + 0.25 * p.x + 0.5 * p.y;
        }
      }

      // Each half has area 1/2. The exact mean fast/thermal fluxes are
      // (2.25, -0.1875) on the left and (2.75, -0.0625) on the right.
      // sigma_f=(.12,.03), chi_prompt=chi_delayed=(1,0), nu_p=1.987 and nu_d=.013.
      // The right material's density is 1.2 times the left before Scale is applied.
      for (const double scale : {1.0, 1.5, 0.75, 1.0})
      {
        SCOPED_TRACE(::testing::Message() << "scale=" << scale);
        right_xs->Scale(scale);
        const double nu = delayed ? 2.0 : 1.987;
        const double left = adjoint ? 0.15 * 2.25 : 0.12 * 2.25 - 0.03 * 0.1875;
        const double right = adjoint ? 0.15 * 2.75 : 0.12 * 2.75 - 0.03 * 0.0625;
        const double expected = 0.5 * nu * (left + 1.2 * scale * right);
        // Affine PWLD integration is exact. This bound covers only ordinary roundoff in local
        // summation, material coefficients, and MPI reductions on 1, 2, and 4 ranks.
        EXPECT_NEAR(ComputeFissionProduction(*problem, phi), expected, 5.0e-13);
      }
    }
}
