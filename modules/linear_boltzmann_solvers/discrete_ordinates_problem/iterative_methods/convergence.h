// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include <optional>
#include <vector>

namespace opensn
{

class LBSProblem;

struct RichardsonConvergenceInfo
{
  double rho = 0.0;
  double phi_tolerance = 0.0;
  double psi_tolerance = 0.0;
  bool psi_check_active = false;
  bool converged = false;
};

double ComputePointwisePhiChange(
  LBSProblem& lbs_problem,
  int groupset_id,
  std::optional<const std::reference_wrapper<std::vector<double>>> opt_phi_old = std::nullopt);

double ComputePointwisePhiChange(
  LBSProblem& lbs_problem,
  std::optional<const std::reference_wrapper<std::vector<double>>> opt_phi_old = std::nullopt);

double ComputePointwisePhiChange(
  LBSProblem& lbs_problem,
  std::vector<int> groupset_ids,
  std::optional<const std::reference_wrapper<std::vector<double>>> opt_phi_old = std::nullopt);

double ComputeL2PhiChange(
  LBSProblem& lbs_problem,
  std::optional<const std::reference_wrapper<std::vector<double>>> opt_phi_old = std::nullopt);

RichardsonConvergenceInfo EvaluateRichardsonConvergence(double phi_change,
                                                        double previous_phi_change,
                                                        double psi_change,
                                                        bool psi_check_active,
                                                        double residual_tolerance);

} // namespace opensn
