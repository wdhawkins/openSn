// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

namespace opensn
{

class DiscreteOrdinatesProblem;
class UnknownManager;
class VectorGhostCommunicator;

/** Evaluates residual terms that use the same angular upwind data as a transport sweep. */
class SweepResidualEvaluator
{
public:
  explicit SweepResidualEvaluator(DiscreteOrdinatesProblem& do_problem);

  void AddStreamingResidual(const UnknownManager& residual_uk_man,
                            const std::vector<double>& phi0,
                            std::vector<double>& residual);

  void AddStreamingResidualComponents(const UnknownManager& residual_uk_man,
                                      const std::vector<double>& phi0,
                                      std::vector<double>& removal,
                                      std::vector<double>& volume_streaming,
                                      std::vector<double>& face_streaming);

private:
  DiscreteOrdinatesProblem& do_problem_;
  std::vector<std::shared_ptr<VectorGhostCommunicator>> psi_ghost_communicators_;
  std::vector<std::vector<double>> ghosted_psi_new_local_;
};

} // namespace opensn
