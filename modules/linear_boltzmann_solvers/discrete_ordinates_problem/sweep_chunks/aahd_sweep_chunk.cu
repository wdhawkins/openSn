// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/aah.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "caliper/cali.h"
#include "caribou/main.hpp"
#include <cstdlib>

namespace crb = caribou;

namespace opensn
{

namespace
{

void
AtomicStoreMax(std::atomic<std::size_t>& target, std::size_t value)
{
  auto current = target.load(std::memory_order_relaxed);
  while (current < value and
         not target.compare_exchange_weak(current,
                                          value,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed))
  {
  }
}

unsigned int
GetAAHSweepThreadBudget()
{
  const char* env_value =
    std::getenv("OPENSN_AAH_SWEEP_THREAD_BUDGET"); // NOLINT(concurrency-mt-unsafe)
  if (env_value != nullptr)
  {
    const auto parsed = std::strtoul(env_value, nullptr, 10);
    if (parsed >= 32)
      return static_cast<unsigned int>(parsed);
  }

#if defined(__HIPCC__)
  return 256;
#else
  return gpu_kernel::threshold;
#endif
}

} // namespace

AAHDSweepChunk::AAHDSweepChunk(DiscreteOrdinatesProblem& problem, LBSGroupset& groupset)
  : SweepChunk(problem.GetPhiNewLocal(),
               problem.GetPsiNewLocal()[groupset.id],
               problem.GetGrid(),
               problem.GetSpatialDiscretization(),
               problem.GetUnitCellMatrices(),
               problem.GetCellTransportViews(),
               problem.GetCellOutflowViews(),
               problem.GetQMomentsLocal(),
               groupset,
               problem.GetBlockID2XSMap(),
               problem.GetNumMoments(),
               problem.GetMaxCellDOFCount(),
               problem.GetMinCellDOFCount()),
    problem_(problem)
{
}

void
AAHDSweepChunk::Sweep(AngleSet& angle_set)
{
  // prepare arguments
  auto& aahd_angle_set = static_cast<AAHD_AngleSet&>(angle_set);
  auto& fluds = static_cast<AAHD_FLUDS&>(aahd_angle_set.GetFLUDS());
  auto& stream = aahd_angle_set.GetStream();
  gpu_kernel::Arguments<gpu_kernel::SweepType::AAH> args(
    problem_, groupset_, aahd_angle_set, fluds, surface_source_active_);
  double* saved_psi = fluds.GetSavedAngularFluxDevicePointer();
  // retrieve SPDS levels
  const auto& spds = static_cast<const AAH_SPDS&>(aahd_angle_set.GetSPDS());
  const auto& levelized_spls = spds.GetLevelizedLocalSubgrid();
  const auto levels_in_angle_set = levelized_spls.size();
  angle_set_sweep_count_.fetch_add(1, std::memory_order_relaxed);
  total_level_count_.fetch_add(levels_in_angle_set, std::memory_order_relaxed);
  AtomicStoreMax(max_levels_per_angle_set_, levels_in_angle_set);
  // compute block size
  const unsigned int thread_budget = GetAAHSweepThreadBudget();
  unsigned int stride_size =
    gpu_kernel::RoundUp(static_cast<unsigned int>(args.flud_data.stride_size));
  unsigned int block_size_x = std::min(stride_size, thread_budget);
  unsigned int block_size_y = std::max(1U, thread_budget / block_size_x);
  crb::Dim3 block_size(block_size_x, block_size_y);
  unsigned int grid_size_x = (stride_size + thread_budget - 1) / thread_budget;
  for (std::uint32_t level = 0; level < levelized_spls.size(); ++level)
  {
    // compute grid size
    std::size_t level_size = levelized_spls[level].size();
    actual_kernel_launch_count_.fetch_add(1, std::memory_order_relaxed);
    total_level_cell_count_.fetch_add(level_size, std::memory_order_relaxed);
    AtomicStoreMax(max_level_cell_count_, level_size);
    unsigned int grid_size_y = (level_size + block_size_y - 1) / block_size_y;
    crb::Dim3 grid_size(grid_size_x, grid_size_y);
    // perform the sweep on device
    const std::uint32_t* level_data = spds.GetDeviceLevelVector(level);
#if defined(__NVCC__) || defined(__HIPCC__)
    gpu_kernel::SweepKernel<gpu_kernel::SweepType::AAH><<<grid_size, block_size, 0, stream>>>(
      args, level_data, static_cast<unsigned int>(level_size), saved_psi);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    stream.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<3> work_index)
                        {
                          gpu_kernel::SweepKernel<gpu_kernel::SweepType::AAH>(
                            args, level_data, static_cast<unsigned int>(level_size), saved_psi);
                        });
#endif
  }
}

void
AAHDSweepChunk::ResetLaunchProfile()
{
  angle_set_sweep_count_.store(0, std::memory_order_relaxed);
  actual_kernel_launch_count_.store(0, std::memory_order_relaxed);
  total_level_count_.store(0, std::memory_order_relaxed);
  max_levels_per_angle_set_.store(0, std::memory_order_relaxed);
  total_level_cell_count_.store(0, std::memory_order_relaxed);
  max_level_cell_count_.store(0, std::memory_order_relaxed);
}

AAHDSweepChunk::LaunchProfile
AAHDSweepChunk::GetLaunchProfile() const
{
  return {.angle_set_sweeps = angle_set_sweep_count_.load(std::memory_order_relaxed),
          .actual_kernel_launches = actual_kernel_launch_count_.load(std::memory_order_relaxed),
          .total_levels = total_level_count_.load(std::memory_order_relaxed),
          .max_levels_per_angle_set = max_levels_per_angle_set_.load(std::memory_order_relaxed),
          .total_level_cells = total_level_cell_count_.load(std::memory_order_relaxed),
          .max_level_cells = max_level_cell_count_.load(std::memory_order_relaxed)};
}

} // namespace opensn
