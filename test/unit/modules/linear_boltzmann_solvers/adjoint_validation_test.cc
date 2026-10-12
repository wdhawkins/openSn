// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/materials/multi_group_xs/multi_group_xs.h"
#include "framework/math/quadratures/angular/curvilinear_product_quadrature.h"
#include "framework/math/quadratures/angular/product_quadrature.h"
#include "framework/parameters/parameter_block.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_curvilinear_problem/discrete_ordinates_curvilinear_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/postprocessors/cross_section_sensitivity_postprocessor.h"
#include "test/unit/common/mesh_builders.h"
#include "gtest/gtest.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace opensn;

namespace
{

ParameterBlock
MakeProblemParameters(const std::shared_ptr<MeshContinuum>& grid,
                      const std::shared_ptr<AngularQuadrature>& quadrature,
                      const std::string& angle_aggregation_type,
                      bool adjoint)
{
  ParameterBlock groupset("0");
  groupset.AddParameter("groups_from_to", std::vector<unsigned int>{0, 0});
  groupset.AddParameter("angular_quadrature", quadrature);
  groupset.AddParameter("angle_aggregation_type", angle_aggregation_type);

  ParameterBlock groupsets("groupsets");
  groupsets.ChangeToArray();
  groupsets.AddParameter(std::move(groupset));

  auto xs = std::make_shared<MultiGroupXS>(MultiGroupXS::CreateSimpleOneGroup(1.0, 0.5));
  ParameterBlock xs_entry("0");
  xs_entry.AddParameter("block_ids", std::vector<unsigned int>{0});
  xs_entry.AddParameter("xs", xs);

  ParameterBlock xs_map("xs_map");
  xs_map.ChangeToArray();
  xs_map.AddParameter(std::move(xs_entry));

  ParameterBlock options("options");
  options.AddParameter("adjoint", adjoint);

  ParameterBlock params;
  params.AddParameter("mesh", grid);
  params.AddParameter("num_groups", 1U);
  params.AddParameter(std::move(groupsets));
  params.AddParameter(std::move(xs_map));
  params.AddParameter(std::move(options));
  return params;
}

std::shared_ptr<DiscreteOrdinatesProblem>
MakeCartesianProblem(OperatorConstructionMethod method, bool adjoint)
{
  auto grid = BuildSquareMesh(1.0, 1, 0.0);
  grid->SetUniformBlockID(0);
  auto quadrature = std::make_shared<GLCProductQuadrature2DXY>(2, 4, 1, false, method);
  return DiscreteOrdinatesProblem::Create(
    MakeProblemParameters(grid, quadrature, "single", adjoint));
}

std::shared_ptr<DiscreteOrdinatesCurvilinearProblem>
MakeCurvilinearProblem(bool adjoint)
{
  auto grid = BuildSquareMesh(1.0, 1, 0.0);
  grid->SetCoordinateSystem(CoordinateSystemType::CYLINDRICAL);
  grid->SetUniformBlockID(0);
  auto quadrature = std::make_shared<GLCProductQuadrature2DRZ>(2, 4, 0);
  return DiscreteOrdinatesCurvilinearProblem::Create(
    MakeProblemParameters(grid, quadrature, "azimuthal", adjoint));
}

} // namespace

TEST(AdjointValidationTest, RejectsCurvilinearGeometry)
{
  EXPECT_THROW(static_cast<void>(MakeCurvilinearProblem(true)), std::invalid_argument);

  const auto problem = MakeCurvilinearProblem(false);
  EXPECT_THROW(problem->SetAdjoint(true), std::invalid_argument);
}

TEST(AdjointValidationTest, RejectsGalerkinOperators)
{
  EXPECT_NO_THROW(
    static_cast<void>(MakeCartesianProblem(OperatorConstructionMethod::STANDARD, true)));

  for (const auto method :
       {OperatorConstructionMethod::GALERKIN_ONE, OperatorConstructionMethod::GALERKIN_THREE})
  {
    EXPECT_THROW(static_cast<void>(MakeCartesianProblem(method, true)), std::invalid_argument);

    const auto problem = MakeCartesianProblem(method, false);
    EXPECT_THROW(problem->SetAdjoint(true), std::invalid_argument);

    ParameterBlock sensitivity_params;
    sensitivity_params.AddParameter("problem", std::static_pointer_cast<Problem>(problem));
    sensitivity_params.AddParameter("sensitivity_type", "sigma_t");
    EXPECT_THROW(
      static_cast<void>(CrossSectionSensitivityPostprocessor::Create(sensitivity_params)),
      std::invalid_argument);
  }
}
