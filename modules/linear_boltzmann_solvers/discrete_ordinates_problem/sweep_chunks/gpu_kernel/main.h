// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/buffer.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/solver.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "caribou/main.hpp"
#if defined(__NVCC__)
#include <cooperative_groups.h>
#elif defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#endif
#include <utility>
#include <type_traits>

namespace crb = caribou;

namespace opensn::gpu_kernel
{

template <std::uint32_t... I, class F>
__CRB_DEVICE_FUNC__ void
ForDOFs1ToN(std::integer_sequence<std::uint32_t, I...>, F&& f)
{
  (f(std::integral_constant<std::uint32_t, I + 1>{}), ...);
}

template <class F>
__CRB_DEVICE_FUNC__ void
ForDOFs1ToMax(F&& f)
{
  ForDOFs1ToN(std::make_integer_sequence<std::uint32_t, LBSProblem::max_dofs_gpu>{},
              std::forward<F>(f));
}

template <SweepType t, class... Args>
__CRB_DEVICE_FUNC__ void
SweepDispatch(std::uint32_t n, Args&&... args)
{
  bool done = false;
  ForDOFs1ToMax(
    [&](auto dof_c)
    {
      constexpr std::uint32_t dof = decltype(dof_c)::value;
      if (!done && n == dof)
      {
        gpu_kernel::Sweep<dof, t>(std::forward<Args>(args)...);
        done = true;
      }
    });
}

// Per-cell sweep body, factored out so both SweepKernel and SweepPersistentKernel share it.
// Caller must check bounds (cell_idx < num_cells, angle_group_idx < stride_size) before calling.
// level_node_data may be nullptr only for the CBC fallback path (uses GetCellDataIndex instead).
template <SweepType t>
__CRB_DEVICE_FUNC__ void
DoSweepCell(Arguments<t> args,
            unsigned int cell_idx,
            unsigned int angle_group_idx,
            const std::uint32_t* cells_to_sweep,
            const std::uint64_t* level_node_data,
            const std::uint32_t* level_cell_starts,
            std::uint32_t level_abs_start,
            double* saved_psi)
{
  unsigned int angle_idx = angle_group_idx / args.groupset_size;
  unsigned int group_idx = angle_group_idx - angle_idx * args.groupset_size;
  const std::uint32_t cell_local_idx = cells_to_sweep[cell_idx];
  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, cell_local_idx);
  if (cell.num_nodes == 0)
    return;
  const std::uint64_t* cell_edge_data;
  if (level_node_data != nullptr)
    cell_edge_data = level_node_data + level_cell_starts[level_abs_start + cell_idx];
  else
  {
    auto [ced, ignored] = GetCellDataIndex(args.flud_index, cell_local_idx);
    cell_edge_data = ced;
  }
  std::uint32_t num_moments;
  std::uint32_t direction_num = args.directions[angle_idx];
  DirectionView direction;
  {
    QuadratureView quadrature(args.quad_data);
    num_moments = quadrature.num_moments;
    quadrature.GetDirectionView(direction, direction_num);
  }
  opensn::gpu_kernel::SweepDispatch<t>(cell.num_nodes,
                                       args,
                                       cell,
                                       direction,
                                       cell_edge_data,
                                       angle_group_idx,
                                       group_idx,
                                       num_moments,
                                       saved_psi);
}

// Per-level sweep kernel: one launch per level, grid sized to that level's cell count.
template <SweepType t>
__CRB_GLOBAL_FUNC__ void
SweepKernel(Arguments<t> args,
            const std::uint32_t* cells_to_sweep,
            unsigned int num_cells,
            const std::uint64_t* level_node_data,
            const std::uint32_t* level_cell_starts,
            std::uint32_t level_abs_start,
            double* saved_psi)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  unsigned int cell_idx = threadIdx.y + blockDim.y * blockIdx.y;
  unsigned int angle_group_idx = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  unsigned int cell_idx = work_index.get_global_id(1);
  unsigned int angle_group_idx = work_index.get_global_id(2);
#endif
  if (cell_idx >= num_cells || angle_group_idx >= args.flud_data.stride_size)
    return;
  DoSweepCell(
    args, cell_idx, angle_group_idx, cells_to_sweep,
    level_node_data, level_cell_starts, level_abs_start, saved_psi);
}

#if defined(__NVCC__) || defined(__HIPCC__)
// Persistent cooperative kernel: one launch per sweep pass per angle set.
// Loops over all levels internally, using grid.sync() between levels to enforce
// data dependencies.  Eliminates per-level GPU command-processor dispatch overhead
// (~15 μs/kernel × 167 levels) at the cost of one grid.sync() per level transition.
//
// Requires hipLaunchCooperativeKernel / cudaLaunchCooperativeKernel; the grid must
// fit entirely within the GPU's simultaneous resident-block capacity.
//
// cells_flat      : device_levelized_spls_ (all levels concatenated)
// level_offsets   : prefix sums of level sizes — serves as both the per-level offset
//                   into cells_flat AND as level_abs_start for level_cell_starts indexing
//                   (identical to AAHD_FLUDSCommonData::level_abs_starts_)
// level_sizes     : cell count per level
// level_node_data : level-traversal-ordered face node data (from FLUDS common data)
// level_cell_starts: indexed by (level_offsets[level] + cell_idx_within_level)
template <SweepType t>
__CRB_GLOBAL_FUNC__ void
SweepPersistentKernel(Arguments<t> args,
                      const std::uint32_t* cells_flat,
                      const std::uint32_t* level_offsets,
                      const std::uint32_t* level_sizes,
                      std::uint32_t num_levels,
                      const std::uint64_t* level_node_data,
                      const std::uint32_t* level_cell_starts,
                      double* saved_psi)
{
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  const unsigned int cell_idx = threadIdx.y + blockDim.y * blockIdx.y;
  const unsigned int angle_group_idx = threadIdx.x + blockDim.x * blockIdx.x;
  const bool angle_active = (angle_group_idx < args.flud_data.stride_size);

  for (std::uint32_t level = 0; level < num_levels; ++level)
  {
    const std::uint32_t num_cells = level_sizes[level];
    const std::uint32_t level_offset = level_offsets[level]; // = abs start for both arrays
    if (angle_active && cell_idx < num_cells)
      DoSweepCell(args, cell_idx, angle_group_idx,
                  cells_flat + level_offset,
                  level_node_data, level_cell_starts, level_offset,
                  saved_psi);
    grid.sync();
  }
}
#endif

} // namespace opensn::gpu_kernel
