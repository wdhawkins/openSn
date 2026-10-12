// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/quadratures/angular/harmonic_selection_rules.h"
#include "gtest/gtest.h"
#include <cmath>
#include <set>

using namespace opensn;

namespace
{

using Harmonic = AngularQuadrature::HarmonicIndices;
using Parameters = HarmonicSelectionRules::SelectionParameters;

std::vector<std::pair<unsigned int, int>>
AsPairs(const std::vector<Harmonic>& harmonics)
{
  std::vector<std::pair<unsigned int, int>> pairs;
  pairs.reserve(harmonics.size());
  for (const auto& harmonic : harmonics)
    pairs.emplace_back(harmonic.ell, harmonic.m);
  return pairs;
}

void
ExpectValidUniqueHarmonics(const std::vector<Harmonic>& harmonics, const std::size_t expected_size)
{
  ASSERT_EQ(harmonics.size(), expected_size);
  std::set<std::pair<unsigned int, int>> unique_harmonics;
  for (const auto& harmonic : harmonics)
  {
    EXPECT_LE(std::abs(harmonic.m), static_cast<int>(harmonic.ell));
    unique_harmonics.emplace(harmonic.ell, harmonic.m);
  }
  EXPECT_EQ(unique_harmonics.size(), harmonics.size());
}

} // namespace

TEST(HarmonicSelectionRulesTest, StandardRulesMatchSpatialDimension)
{
  Parameters params;
  params.construction_method = OperatorConstructionMethod::STANDARD;
  params.scattering_order = 2;

  params.dimension = 1;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
            (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, 0}, {2, 0}}));

  params.dimension = 2;
  EXPECT_EQ(
    AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
    (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, -1}, {1, 1}, {2, -2}, {2, 0}, {2, 2}}));

  params.dimension = 3;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
            (std::vector<std::pair<unsigned int, int>>{
              {0, 0}, {1, -1}, {1, 0}, {1, 1}, {2, -2}, {2, -1}, {2, 0}, {2, 1}, {2, 2}}));
}

TEST(HarmonicSelectionRulesTest, OneDimensionalGalerkinRulesRespectConstructionMethod)
{
  Parameters params;
  params.quadrature_type = AngularQuadratureType::PRODUCT_QUADRATURE;
  params.dimension = 1;
  params.scattering_order = 2;
  params.num_angles = 4;

  params.construction_method = OperatorConstructionMethod::GALERKIN_ONE;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
            (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, 0}, {2, 0}, {3, 0}}));

  params.construction_method = OperatorConstructionMethod::GALERKIN_THREE;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
            (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, 0}, {2, 0}}));
}

TEST(HarmonicSelectionRulesTest, ProductGalerkinRulesSelectDocumentedMomentSets)
{
  Parameters params;
  params.quadrature_type = AngularQuadratureType::PRODUCT_QUADRATURE;
  params.construction_method = OperatorConstructionMethod::GALERKIN_ONE;
  params.scattering_order = 2;
  params.num_angles = 16;
  params.n_polar = 4;
  params.n_azimuthal = 8;

  params.dimension = 2;
  EXPECT_EQ(
    AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
    (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, -1}, {1, 1}, {2, -2}, {2, 2}, {2, 0}}));

  params.dimension = 3;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
            (std::vector<std::pair<unsigned int, int>>{
              {0, 0}, {1, 0}, {1, 1}, {1, -1}, {2, 0}, {2, 1}, {2, -1}, {2, 2}, {2, -2}}));
}

TEST(HarmonicSelectionRulesTest, TriangularGalerkinRulesFillIndependentMomentSets)
{
  Parameters params;
  params.quadrature_type = AngularQuadratureType::TRIANGULAR_QUADRATURE;
  params.construction_method = OperatorConstructionMethod::GALERKIN_THREE;
  params.scattering_order = 20;

  params.dimension = 2;
  params.num_angles = 8;
  ExpectValidUniqueHarmonics(HarmonicSelectionRules::SelectHarmonics(params), 8);

  params.dimension = 3;
  params.num_angles = 16;
  ExpectValidUniqueHarmonics(HarmonicSelectionRules::SelectHarmonics(params), 16);
}

TEST(HarmonicSelectionRulesTest, EmpiricalRulesInferOrderAndTruncateScatteringOrder)
{
  Parameters params;
  params.construction_method = OperatorConstructionMethod::GALERKIN_ONE;
  params.scattering_order = 2;

  params.quadrature_type = AngularQuadratureType::LEBEDEV_QUADRATURE;
  params.dimension = 3;
  params.num_angles = 6;
  EXPECT_EQ(
    AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
    (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, -1}, {1, 0}, {1, 1}, {2, 0}, {2, 2}}));

  params.quadrature_type = AngularQuadratureType::SLDFE_SQ;
  params.dimension = 2;
  params.num_angles = 48;
  EXPECT_EQ(
    AsPairs(HarmonicSelectionRules::SelectHarmonics(params)),
    (std::vector<std::pair<unsigned int, int>>{{0, 0}, {1, -1}, {1, 1}, {2, -2}, {2, 0}, {2, 2}}));
}

TEST(HarmonicSelectionRulesTest, UnknownEmpiricalOrdersFallBackToStandardRules)
{
  Parameters params;
  params.construction_method = OperatorConstructionMethod::GALERKIN_ONE;
  params.dimension = 3;
  params.scattering_order = 1;
  params.num_angles = 7;

  const std::vector<std::pair<unsigned int, int>> standard_order_one = {
    {0, 0}, {1, -1}, {1, 0}, {1, 1}};

  params.quadrature_type = AngularQuadratureType::LEBEDEV_QUADRATURE;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)), standard_order_one);

  params.quadrature_type = AngularQuadratureType::SLDFE_SQ;
  params.num_angles = 97;
  EXPECT_EQ(AsPairs(HarmonicSelectionRules::SelectHarmonics(params)), standard_order_one);
}

TEST(HarmonicSelectionRulesTest, MissingQuadratureTypeIsRejected)
{
  Parameters params;
  params.construction_method = OperatorConstructionMethod::GALERKIN_ONE;
  params.dimension = 3;
  params.scattering_order = 1;
  params.num_angles = 4;

  EXPECT_THROW(HarmonicSelectionRules::SelectHarmonics(params), std::logic_error);
}
