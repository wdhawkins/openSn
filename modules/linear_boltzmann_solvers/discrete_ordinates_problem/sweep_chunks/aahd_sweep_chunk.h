// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include <atomic>
#include <cstdint>
#include <array>
#include <cmath>

namespace opensn
{

class DiscreteOrdinatesProblem;

class AAHDSweepChunk : public SweepChunk
{
public:
  struct LaunchProfile
  {
    std::size_t angle_set_sweeps = 0;
    std::size_t actual_kernel_launches = 0;
    std::size_t total_levels = 0;
    std::size_t max_levels_per_angle_set = 0;
    std::size_t total_level_cells = 0;
    std::size_t max_level_cells = 0;
  };

  AAHDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset);

  DiscreteOrdinatesProblem& GetProblem() { return problem_; }
  MeshContinuum& GetGrid() { return *grid_; }
  const LBSGroupset& GetGroupset() const { return groupset_; }

  void Sweep(AngleSet& angle_set) override;
  void ResetLaunchProfile();
  LaunchProfile GetLaunchProfile() const;

protected:
  DiscreteOrdinatesProblem& problem_;
  std::atomic<std::size_t> angle_set_sweep_count_{0};
  std::atomic<std::size_t> actual_kernel_launch_count_{0};
  std::atomic<std::size_t> total_level_count_{0};
  std::atomic<std::size_t> max_levels_per_angle_set_{0};
  std::atomic<std::size_t> total_level_cell_count_{0};
  std::atomic<std::size_t> max_level_cell_count_{0};
};

} // namespace opensn
