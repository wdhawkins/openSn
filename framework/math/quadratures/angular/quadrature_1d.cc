// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/quadratures/angular/quadrature_1d.h"
#include "framework/math/quadratures/gausslegendre_quadrature.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <cmath>
#include <sstream>
#include <cassert>

namespace opensn
{

void
Quadrature1D::AssemblePolarCosines(const std::vector<double>& polar,
                                   const std::vector<double>& wts,
                                   bool verbose)
{
  size_t Np = polar.size();

  polar_ang_ = polar;

  if (verbose)
  {
    log.Log() << "Polar angles:";
    for (const auto& ang : polar_ang_)
      log.Log() << ang;
  }

  // Create angle pairs
  map_directions_.clear();

  abscissae_.clear();
  weights_.clear();
  std::stringstream ostr;
  weight_sum_ = 0.0;
  for (unsigned int j = 0; j < Np; ++j)
  {
    map_directions_.emplace(j, std::vector<unsigned int>{j});

    const auto abscissa = QuadraturePointPhiTheta(0.0, polar_ang_[j]);

    abscissae_.emplace_back(abscissa);

    const double weight = wts[j];
    weights_.emplace_back(weight);
    weight_sum_ += weight;

    if (verbose)
    {
      ostr << "Varphi=" << std::fixed << std::setprecision(2) << abscissa.phi * 180.0 / M_PI
           << " Theta=" << std::fixed << std::setprecision(2) << abscissa.theta * 180.0 / M_PI
           << " Weight=" << std::scientific << std::setprecision(3) << weight << '\n';
    }
  }

  // Create omega list
  omegas_.clear();
  for (const auto& qpoint : abscissae_)
  {
    Vector3 new_omega;
    new_omega.x = sin(qpoint.theta) * cos(qpoint.phi);
    new_omega.y = sin(qpoint.theta) * sin(qpoint.phi);
    new_omega.z = cos(qpoint.theta);

    omegas_.emplace_back(new_omega);

    if (verbose)
      log.Log() << "Quadrature angle=" << new_omega.PrintStr();
  }

  // Normalize weights to 1.0
  for (auto& w : weights_)
    w /= weight_sum_;

  // Compute and store sum of weights after normalization
  weight_sum_ = 0.0;
  for (auto& w : weights_)
    weight_sum_ += w;
}

GLQuadrature1DSlab::GLQuadrature1DSlab(unsigned int Npolar,
                                       unsigned int scattering_order,
                                       bool verbose,
                                       OperatorConstructionMethod method)
  : Quadrature1D(1, scattering_order, method)
{
  if (Npolar % 2 != 0)
    throw std::invalid_argument("GLQuadrature1DSlab: Npolar must be even.");

  n_polar_ = Npolar;

  GaussLegendreQuadrature gl_polar(Npolar);

  // Create polar angles
  polar_ang_.clear();
  for (unsigned int j = 0; j < Npolar; ++j)
    polar_ang_.emplace_back(M_PI - acos(gl_polar.qpoints[j][0]));

  // Create weights
  auto& weights = gl_polar.weights;

  // Initialize
  AssemblePolarCosines(polar_ang_, weights, verbose);
  MakeHarmonicIndices();
  BuildDiscreteToMomentOperator();
  BuildMomentToDiscreteOperator();
}

} // namespace opensn
