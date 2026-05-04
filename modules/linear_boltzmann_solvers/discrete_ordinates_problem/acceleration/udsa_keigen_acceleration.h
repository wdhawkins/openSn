// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/discrete_ordinates_keigen_acceleration.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/udsa.h"

namespace opensn
{

/** UDSA acceleration for power-iteration k-eigenvalue solves. */
class UDSAKEigenAcceleration : public DiscreteOrdinatesKEigenAcceleration
{
public:
  explicit UDSAKEigenAcceleration(const InputParameters& params);

  void Initialize() final;
  void PreExecute() final;
  void PrePowerIteration() final;
  double PostPowerIteration() final;

private:
  void BuildFissionSource(const std::vector<double>& phi0,
                          double lambda,
                          std::vector<double>& q0) const;

  double production_ell_ = 1.0;

  UDSADiffusionAcceleration diffusion_acceleration_;

  std::vector<double> phi0_;
  std::vector<double> phi0_ell_;
  std::vector<double> phi0_star_;
  std::vector<double> phi0_m_;
  std::vector<double> phi0_kp1_;
  std::vector<double> q0_;
  std::vector<double> boundary_source_;
  std::vector<double> current_correction_;
  std::vector<double> sweep_source_;
  std::vector<double> sweep_rhs_;
  std::vector<double> operator_phi_;
  std::vector<double> phi_temp_;

public:
  static InputParameters GetInputParameters();

  static std::shared_ptr<UDSAKEigenAcceleration> Create(const ParameterBlock& params);
};

} // namespace opensn
