// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/quadratures/angular/angular_quadrature.h"
#include <map>
#include <vector>

namespace opensn
{

/// Base class for 1D angular quadratures.
///
/// Unlike product quadratures, 1D quadratures have no azimuthal degree of freedom: every
/// direction lies at azimuthal angle phi=0, so the quadrature is purely a function of the polar
/// angle.
class Quadrature1D : public AngularQuadrature
{
public:
  ~Quadrature1D() override = default;

  /// Return the abscissae index for the given polar angle index.
  unsigned int GetAngleNum(const unsigned int polar_angle_index) const
  {
    return map_directions_.at(polar_angle_index).front();
  }

  /// Return constant reference to map_directions.
  const std::map<unsigned int, std::vector<unsigned int>>& GetDirectionMap() const
  {
    return map_directions_;
  }

  const std::vector<double>& GetPolarAngles() const { return polar_ang_; }

  unsigned int GetNumPolarAngles() const override { return n_polar_; }

  unsigned int GetNumAzimuthalAngles() const override { return 1; }

protected:
  Quadrature1D(unsigned int dimension,
              unsigned int scattering_order,
              OperatorConstructionMethod method)
    : AngularQuadrature(AngularQuadratureType::POLAR_QUADRATURE, dimension, scattering_order, method),
      weight_sum_(0.0)
  {
  }

  /// Initializes the quadrature with custom polar angles and weights. Every abscissa is placed
  /// at azimuthal angle phi=0.
  void AssemblePolarCosines(const std::vector<double>& polar,
                            const std::vector<double>& wts,
                            bool verbose);

  double weight_sum_;

  /// Polar angles for this quadrature.
  std::vector<double> polar_ang_;

  /// Linear indices of ordered directions mapped to polar level. Each polar level maps to
  /// exactly one direction.
  std::map<unsigned int, std::vector<unsigned int>> map_directions_;

  /// Number of polar angles specified for this quadrature.
  unsigned int n_polar_ = 0;
};

class GLQuadrature1DSlab : public Quadrature1D
{
public:
  /// Constructor for 1D slab Gauss-Legendre quadrature
  explicit GLQuadrature1DSlab(
    unsigned int Npolar,
    unsigned int scattering_order,
    bool verbose = false,
    OperatorConstructionMethod method = OperatorConstructionMethod::STANDARD);

  std::string GetName() const override { return "1D Slab Gauss-Legendre"; }
};

} // namespace opensn
