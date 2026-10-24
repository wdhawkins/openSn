// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/aahd_batch_slot.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/arguments.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "caribou/main.hpp"

namespace crb = caribou;

namespace opensn::gpu_kernel
{

/**
 * Call a function with the cell DOF count as a compile-time constant.
 * With `fixed_ndofs > 0` only that count is compiled. This keeps register and stack use at what
 * that cell type needs. The caller guarantees that every cell has that count. With
 * `fixed_ndofs == 0` every count up to LBSProblem::max_dofs_gpu is compiled and selected at run
 * time.
 *
 * \tparam fixed_ndofs Single compiled cell DOF count, or 0 to compile all counts.
 * \tparam F Type of the function.
 * \param num_nodes Number of nodes (DOFs) of the cell.
 * \param f Function called with a `std::integral_constant` of the DOF count.
 */
template <std::uint32_t fixed_ndofs, class F>
__CRB_DEVICE_FUNC__ void
DispatchCellDOFs(std::uint32_t num_nodes, F& f)
{
  if constexpr (fixed_ndofs > 0)
  {
    if (num_nodes == fixed_ndofs)
      f(std::integral_constant<std::uint32_t, fixed_ndofs>{});
  }
  else
  {
    bool done = false;
    auto dispatch = [&](auto dof_c)
    {
      if (!done && num_nodes == decltype(dof_c)::value)
      {
        f(dof_c);
        done = true;
      }
    };
    ForDOFs1ToMax(dispatch);
  }
}

/**
 * Assemble a thread's sweep arguments from the shared launch data and its batch slot.
 *
 * \param shared_args Launch data shared by every angle set of the launch.
 * \param slot Batch slot of the thread's angle set.
 * \return Arguments of the thread's angle set.
 */
__CRB_DEVICE_FUNC__ Arguments<SweepKind::AAH>
SlotArguments(const Arguments<SweepKind::AAH>& shared_args, const AAHD_BatchSlot& slot)
{
  Arguments<SweepKind::AAH> args = shared_args;
  args.directions = slot.directions;
  args.boundary_offset = slot.boundary_offset;
  args.angleset_size = slot.angleset_size;
  args.flud_index = slot.flud_index;
  args.flud_data = slot.flud_data;
  return args;
}

/**
 * Sweep one (cell, angle-group) pair of a batch slot's level.
 *
 * \tparam fixed_ndofs Single compiled cell DOF count, or 0 for all counts (see DispatchCellDOFs).
 * \param shared_args Launch data shared by every angle set of the launch.
 * \param slot Batch slot of the angle set's level.
 * \param cell_idx Index of the cell in the level.
 * \param angle_group_idx Angle-group index within the angle set.
 */
template <std::uint32_t fixed_ndofs>
__CRB_DEVICE_FUNC__ void
SweepSlotCell(const Arguments<SweepKind::AAH>& shared_args,
              const AAHD_BatchSlot& slot,
              unsigned int cell_idx,
              unsigned int angle_group_idx)
{
  const Arguments<SweepKind::AAH> args = SlotArguments(shared_args, slot);
  unsigned int angle_idx = angle_group_idx / args.groupset_size;
  unsigned int group_idx = angle_group_idx - angle_idx * args.groupset_size;
  const std::uint32_t cell_local_idx = slot.cells_to_sweep[cell_idx];
  CellView cell;
  MeshView(args.mesh_data).GetCellView(cell, cell_local_idx);
  if (cell.num_nodes == 0)
    return;
  auto [cell_edge_data, _] = GetCellDataIndex(args.flud_index, cell_local_idx);
  std::uint32_t num_moments;
  DirectionView direction;
  {
    QuadratureView quadrature(args.quad_data);
    num_moments = quadrature.num_moments;
    quadrature.GetDirectionView(direction, args.directions[angle_idx]);
  }
  auto sweep = [&](auto dof_c)
  {
    Sweep<decltype(dof_c)::value, SweepKind::AAH>(args,
                                                  cell,
                                                  direction,
                                                  cell_edge_data,
                                                  angle_group_idx,
                                                  group_idx,
                                                  num_moments,
                                                  slot.saved_psi);
  };
  DispatchCellDOFs<fixed_ndofs>(cell.num_nodes, sweep);
}

/**
 * Sweep the current level of every batch slot, with one thread per cell and angle-group.
 * Each grid-z slice (blockIdx.z, or SYCL dimension 0) is a different angle set's current level.
 * One launch therefore covers many angle sets instead of one. The per-thread work is the same
 * solver.h Sweep as `SweepKernel` in main.h. Its arguments are assembled from the shared launch
 * data and the thread's batch slot.
 *
 * \tparam fixed_ndofs Single compiled cell DOF count, or 0 for all counts (see DispatchCellDOFs).
 * \param shared_args Launch data shared by every angle set of the launch.
 * \param batch_slots Batch slots of the launch, one per grid-z slice.
 * \param linear_mapping If true, x enumerates the slot's (cell, angle-group) pairs cell by cell,
 *        so a warp covers consecutive cells' angle-groups with no idle lanes. If false, x is the
 *        angle-group and y is the cell, which spreads a small level over more warps.
 */
template <std::uint32_t fixed_ndofs>
__CRB_GLOBAL_FUNC__ void
AAHDFusedSweepKernel(Arguments<SweepKind::AAH> shared_args,
                     const AAHD_BatchSlot* batch_slots,
                     bool linear_mapping)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  unsigned int slot_idx = blockIdx.z;
  unsigned int x_idx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int y_idx = threadIdx.y + blockDim.y * blockIdx.y;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  unsigned int slot_idx = work_index.get_global_id(0);
  unsigned int x_idx = work_index.get_global_id(2);
  unsigned int y_idx = work_index.get_global_id(1);
#endif
  const AAHD_BatchSlot& slot = batch_slots[slot_idx];
  const unsigned int slot_stride = shared_args.groupset_size * slot.angleset_size;
  unsigned int cell_idx = linear_mapping ? x_idx / slot_stride : y_idx;
  unsigned int angle_group_idx = linear_mapping ? x_idx - cell_idx * slot_stride : x_idx;
  if (cell_idx >= slot.num_cells || angle_group_idx >= slot_stride)
    return;
  SweepSlotCell<fixed_ndofs>(shared_args, slot, cell_idx, angle_group_idx);
}

/**
 * Sweep one level of every batch slot, with the slots' (cell, angle-group) pairs flattened into one
 * index range. Slot `s` owns the indices from `batch_slots[s].work_offset` to the next slot's
 * offset, so slots with small levels launch no idle threads. A thread finds its slot by binary
 * search over the offsets.
 *
 * \tparam fixed_ndofs Single compiled cell DOF count, or 0 for all counts (see DispatchCellDOFs).
 * \param shared_args Launch data shared by every angle set of the launch.
 * \param batch_slots Batch slots of the launch, in increasing work_offset order.
 * \param num_slots Number of slots.
 * \param total_work Total number of (cell, angle-group) pairs.
 */
template <std::uint32_t fixed_ndofs>
__CRB_GLOBAL_FUNC__ void
AAHDFusedFlatKernel(Arguments<SweepKind::AAH> shared_args,
                    const AAHD_BatchSlot* batch_slots,
                    unsigned int num_slots,
                    unsigned int total_work)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const unsigned int idx = work_index.get_global_id(2);
#endif
  if (idx >= total_work)
    return;
  unsigned int lo = 0;
  unsigned int hi = num_slots - 1;
  while (lo < hi)
  {
    const unsigned int mid = (lo + hi + 1) / 2;
    if (batch_slots[mid].work_offset <= idx)
      lo = mid;
    else
      hi = mid - 1;
  }
  const AAHD_BatchSlot& slot = batch_slots[lo];
  const unsigned int slot_stride = shared_args.groupset_size * slot.angleset_size;
  const unsigned int local = idx - slot.work_offset;
  const unsigned int cell_idx = local / slot_stride;
  SweepSlotCell<fixed_ndofs>(shared_args, slot, cell_idx, local - cell_idx * slot_stride);
}

/**
 * Sweep every level of every batch slot in one launch, with one block per angle set.
 * The block walks its angle set's levels in order. Its threads stride over each level's
 * (cell, angle-group) pairs, and a block barrier separates the levels, so a level reads the
 * angular fluxes the previous levels wrote. This replaces one launch per level with one launch
 * per run, at the cost of putting each angle set on one compute unit. It suits runs whose levels
 * are small.
 *
 * \tparam fixed_ndofs Single compiled cell DOF count, or 0 for all counts (see DispatchCellDOFs).
 * \param shared_args Launch data shared by every angle set of the launch.
 * \param batch_slots Batch slots of the run, `num_levels` rows of `batch_size` slots.
 * \param batch_size Number of angle sets in the run.
 * \param num_levels Number of levels in the run.
 */
template <std::uint32_t fixed_ndofs>
__CRB_GLOBAL_FUNC__ void
AAHDFusedRunKernel(Arguments<SweepKind::AAH> shared_args,
                   const AAHD_BatchSlot* batch_slots,
                   unsigned int batch_size,
                   unsigned int num_levels)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  const unsigned int member = blockIdx.z;
  const unsigned int tid = threadIdx.x;
  const unsigned int num_threads = blockDim.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const unsigned int member = work_index.get_group(0);
  const unsigned int tid = work_index.get_local_id(2);
  const unsigned int num_threads = work_index.get_local_range(2);
#endif
  for (unsigned int level = 0; level < num_levels; ++level)
  {
    const AAHD_BatchSlot& slot = batch_slots[level * batch_size + member];
    if (slot.cells_to_sweep == nullptr)
      break;
    const unsigned int slot_stride = shared_args.groupset_size * slot.angleset_size;
    const unsigned int work = slot.num_cells * slot_stride;
    for (unsigned int i = tid; i < work; i += num_threads)
    {
      const unsigned int cell_idx = i / slot_stride;
      SweepSlotCell<fixed_ndofs>(shared_args, slot, cell_idx, i - cell_idx * slot_stride);
    }
#if defined(__NVCC__) || defined(__HIPCC__)
    __syncthreads();
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    ::sycl::group_barrier(work_index.get_group());
#endif
  }
}

} // namespace opensn::gpu_kernel
