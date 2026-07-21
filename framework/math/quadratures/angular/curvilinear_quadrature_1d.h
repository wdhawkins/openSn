// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/quadratures/angular/quadrature_1d.h"

namespace opensn
{

/// Base class for curvilinear 1D angular quadratures.
///
/// Extends 1D quadratures with direction-dependent parametrizing factors needed for the
/// curvilinear streaming operator.
class CurvilinearQuadrature1D : public Quadrature1D
{
public:
  const std::vector<double>& GetDiamondDifferenceFactor() const { return fac_diamond_difference_; }

  const std::vector<double>& GetStreamingOperatorFactor() const { return fac_streaming_operator_; }

  ~CurvilinearQuadrature1D() override = default;

protected:
  CurvilinearQuadrature1D(unsigned int dimension,
                          unsigned int scattering_order,
                          OperatorConstructionMethod method)
    : Quadrature1D(dimension, scattering_order, method)
  {
  }

  /// Factor to account for angular diamond differencing.
  std::vector<double> fac_diamond_difference_;

  /// Factor for discretization of the angular-derivative component of the streaming operator.
  std::vector<double> fac_streaming_operator_;
};

class GLQuadrature1DSpherical : public CurvilinearQuadrature1D
{
public:
  GLQuadrature1DSpherical(
    unsigned int Npolar,
    unsigned int scattering_order,
    bool verbose = false,
    OperatorConstructionMethod method = OperatorConstructionMethod::STANDARD);

  ~GLQuadrature1DSpherical() override = default;

  std::string GetName() const override { return "1D Spherical Gauss-Legendre"; }

  void MakeHarmonicIndices();

private:
  /// Initialize with a 1D polar quadrature.
  void Initialize(unsigned int Npolar, bool verbose = false);

  /// Initialize spherical parametrizing factors from the underlying quadrature.
  void InitializeParameters();
};

} // namespace opensn
