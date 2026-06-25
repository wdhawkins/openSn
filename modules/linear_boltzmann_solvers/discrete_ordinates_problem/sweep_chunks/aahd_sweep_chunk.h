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
#if defined(__NVCC__) || defined(__HIPCC__)
#include "caribou/backend.hpp"
#include "caribou/cuhip/api.hpp"
#include <mutex>
#include <unordered_map>
#endif

namespace opensn
{

class DiscreteOrdinatesProblem;
class AAHD_AngleSet;

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
  ~AAHDSweepChunk() override;

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

#if defined(__NVCC__) || defined(__HIPCC__)
  // Per-angle-set captured GPU graphs for the level kernel sequence.
  // On first sweep, the per-level kernel launches are captured into a HIP/CUDA graph.
  // Subsequent sweeps replay the graph with a single dispatch, eliminating per-kernel
  // launch overhead (~14μs/kernel × 4.8M kernels → ~1-5μs/node overhead).
  mutable std::mutex graph_mutex_;
  std::unordered_map<AAHD_AngleSet*, GPU_API(GraphExec_t)> sweep_graphs_;
#endif
};

} // namespace opensn
