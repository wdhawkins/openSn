// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/fused_main.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/gpu_kernel/round_up.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/aah.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "caliper/cali.h"
#include "caribou/main.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>

namespace crb = caribou;

namespace opensn
{

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
  // Each idle lane costs only its stream and slot buffers. Without the fused scheduler only one
  // lane is created. Its stream also identifies the device below.
  lanes_.resize(problem.UseFusedAAO() ? problem.GetFusedAAOLanes() : 1);

  // Fused-kernel launch parameters in units of the device's warp (wavefront) size and
  // compute-unit count, so they scale across NVIDIA, AMD, and Intel GPUs.
  const std::size_t warp_size = crb::get_warp_size();
#if defined(__NVCC__) || defined(__HIPCC__)
  const auto num_compute_units = static_cast<std::size_t>(crb::impl::get_device_attribute(
    crb::impl::get_current_device(),
    ::CUDA_OR_HIP(DevAttrMultiProcessorCount, DeviceAttributeMultiprocessorCount)));
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  const auto num_compute_units = static_cast<std::size_t>(
    lanes_.front().stream.get_device().get_info<::sycl::info::device::max_compute_units>());
#else
  const std::size_t num_compute_units = 1;
#endif
  fused_launch_.linear_block_size = static_cast<unsigned int>(2 * warp_size);
  fused_launch_.min_small_level_stride = static_cast<unsigned int>(warp_size / 4);
  fused_launch_.linear_mapping_min_threads = 4 * warp_size * num_compute_units;
}

void
AAHDSweepChunk::Sweep(AngleSet& angle_set)
{
  // prepare arguments
  auto& aahd_angle_set = dynamic_cast<AAHD_AngleSet&>(angle_set);
  auto& fluds = dynamic_cast<AAHD_FLUDS&>(aahd_angle_set.GetFLUDS());
  auto& stream = aahd_angle_set.GetStream();
  gpu_kernel::Arguments<SweepKind::AAH> args(
    problem_, groupset_, aahd_angle_set, fluds, surface_source_active_);
  double* saved_psi = fluds.GetSavedAngularFluxDevicePointer();
  // retrieve SPDS levels
  const auto& spds = dynamic_cast<const AAH_SPDS&>(aahd_angle_set.GetSPDS());
  const auto& levelized_spls = spds.GetLevelizedLocalSubgrid();
  // compute block size
  unsigned int stride_size =
    gpu_kernel::RoundUp(static_cast<unsigned int>(args.flud_data.stride_size));
  unsigned int block_size_x = std::min(stride_size, gpu_kernel::threshold);
  unsigned int block_size_y = gpu_kernel::threshold / block_size_x;
  crb::Dim3 block_size(block_size_x, block_size_y);
  unsigned int grid_size_x = (stride_size + gpu_kernel::threshold - 1) / gpu_kernel::threshold;
  for (std::uint32_t level = 0; level < levelized_spls.size(); ++level)
  {
    // compute grid size
    std::size_t level_size = levelized_spls[level].size();
    unsigned int grid_size_y = (level_size + block_size_y - 1) / block_size_y;
    crb::Dim3 grid_size(grid_size_x, grid_size_y);
    // perform the sweep on device
    const std::uint32_t* level_data = spds.GetDeviceLevelVector(level);
#if defined(__NVCC__) || defined(__HIPCC__)
    gpu_kernel::SweepKernel<SweepKind::AAH><<<grid_size, block_size, 0, stream>>>(
      args, level_data, static_cast<unsigned int>(level_size), saved_psi);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    stream.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<3> work_index)
                        {
                          gpu_kernel::SweepKernel<SweepKind::AAH>(
                            args, level_data, static_cast<unsigned int>(level_size), saved_psi);
                        });
#endif
  }
}

namespace
{

// Launch one level of a fused run on `stream`.
template <std::uint32_t fixed_ndofs>
void
LaunchFusedLevel(crb::Stream& stream,
                 const gpu_kernel::Arguments<SweepKind::AAH>& shared_args,
                 const gpu_kernel::AAHD_BatchSlot* level_slots,
                 crb::Dim3 grid,
                 crb::Dim3 block,
                 bool linear_mapping)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  gpu_kernel::AAHDFusedSweepKernel<fixed_ndofs>
    <<<grid, block, 0, stream>>>(shared_args, level_slots, linear_mapping);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  stream.parallel_for(
    sycl::nd_range<3>(grid * block, block),
    [=](sycl::nd_item<3> work_index)
    { gpu_kernel::AAHDFusedSweepKernel<fixed_ndofs>(shared_args, level_slots, linear_mapping); });
#endif
}

/// Launch the flat one-level kernel over `total_work` (cell, angle-group) pairs of `count` slots.
void
LaunchFlatLevel(crb::Stream& stream,
                const gpu_kernel::Arguments<SweepKind::AAH>& shared_args,
                const gpu_kernel::AAHD_BatchSlot* slots,
                std::size_t count,
                std::size_t total_work,
                std::uint32_t fixed_ndofs,
                unsigned int block_size)
{
  const auto grid = static_cast<unsigned int>((total_work + block_size - 1) / block_size);
  auto launch = [&](auto fixed_c)
  {
    constexpr std::uint32_t nd = decltype(fixed_c)::value;
#if defined(__NVCC__) || defined(__HIPCC__)
    gpu_kernel::AAHDFusedFlatKernel<nd>
      <<<crb::Dim3(grid, 1, 1), crb::Dim3(block_size, 1), 0, stream>>>(
        shared_args,
        slots,
        static_cast<unsigned int>(count),
        static_cast<unsigned int>(total_work));
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    const auto n = static_cast<unsigned int>(count);
    const auto w = static_cast<unsigned int>(total_work);
    stream.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, grid * block_size), sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3>) { gpu_kernel::AAHDFusedFlatKernel<nd>(shared_args, slots, n, w); });
#endif
  };
  switch (fixed_ndofs)
  {
    case 2:
      launch(std::integral_constant<std::uint32_t, 2>{});
      break;
    case 3:
      launch(std::integral_constant<std::uint32_t, 3>{});
      break;
    case 4:
      launch(std::integral_constant<std::uint32_t, 4>{});
      break;
    case 6:
      launch(std::integral_constant<std::uint32_t, 6>{});
      break;
    case 8:
      launch(std::integral_constant<std::uint32_t, 8>{});
      break;
    default:
      launch(std::integral_constant<std::uint32_t, 0>{});
      break;
  }
}

/**
 * Launch the fused kernel for one level of `count` batch slots.
 * Large levels use the linear (cell, angle-group) mapping, which leaves no idle lanes. Small
 * levels put the angle-groups on x and the cells on y, which spreads their few threads over more
 * warps for latency hiding, unless a cell has too few angle-groups to fill a useful part of a warp.
 * grid-z is the number of slots.
 */
void
LaunchLevelSlots(crb::Stream& stream,
                 const gpu_kernel::Arguments<SweepKind::AAH>& shared_args,
                 const gpu_kernel::AAHD_BatchSlot* slots,
                 std::size_t count,
                 unsigned int max_level_size,
                 unsigned int stride_size,
                 std::uint32_t fixed_ndofs,
                 unsigned int linear_block_size,
                 unsigned int min_small_level_stride,
                 std::size_t linear_mapping_min_threads)
{
  max_level_size = std::max(max_level_size, 1U);
  const std::size_t level_threads = static_cast<std::size_t>(max_level_size) * stride_size * count;
  const bool linear_mapping =
    level_threads >= linear_mapping_min_threads or stride_size < min_small_level_stride;
  crb::Dim3 block_size;
  crb::Dim3 grid_size;
  if (linear_mapping)
  {
    block_size = crb::Dim3(linear_block_size, 1);
    grid_size =
      crb::Dim3((max_level_size * stride_size + linear_block_size - 1) / linear_block_size,
                1,
                static_cast<unsigned int>(count));
  }
  else
  {
    unsigned int block_size_x = std::min(gpu_kernel::RoundUp(stride_size), gpu_kernel::threshold);
    unsigned int block_size_y = gpu_kernel::threshold / block_size_x;
    block_size = crb::Dim3(block_size_x, block_size_y);
    grid_size = crb::Dim3((stride_size + block_size_x - 1) / block_size_x,
                          (max_level_size + block_size_y - 1) / block_size_y,
                          static_cast<unsigned int>(count));
  }
  auto launch = [&](auto fixed_c)
  {
    LaunchFusedLevel<decltype(fixed_c)::value>(
      stream, shared_args, slots, grid_size, block_size, linear_mapping);
  };
  switch (fixed_ndofs)
  {
    case 2:
      launch(std::integral_constant<std::uint32_t, 2>{});
      break;
    case 3:
      launch(std::integral_constant<std::uint32_t, 3>{});
      break;
    case 4:
      launch(std::integral_constant<std::uint32_t, 4>{});
      break;
    case 6:
      launch(std::integral_constant<std::uint32_t, 6>{});
      break;
    case 8:
      launch(std::integral_constant<std::uint32_t, 8>{});
      break;
    default:
      launch(std::integral_constant<std::uint32_t, 0>{});
      break;
  }
}

template <std::uint32_t fixed_ndofs>
void
LaunchFusedRun(crb::Stream& stream,
               const gpu_kernel::Arguments<SweepKind::AAH>& shared_args,
               const gpu_kernel::AAHD_BatchSlot* slots,
               unsigned int batch_size,
               unsigned int num_levels,
               unsigned int block_size)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  gpu_kernel::AAHDFusedRunKernel<fixed_ndofs>
    <<<crb::Dim3(1, 1, batch_size), crb::Dim3(block_size, 1), 0, stream>>>(
      shared_args, slots, batch_size, num_levels);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  stream.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(batch_size, 1, block_size), sycl::range<3>(1, 1, block_size)),
    [=](sycl::nd_item<3> work_index)
    { gpu_kernel::AAHDFusedRunKernel<fixed_ndofs>(shared_args, slots, batch_size, num_levels); });
#endif
}

/// Prototype settings of the one-launch run kernel, read once from the environment.
struct PersistKnobs
{
  /// OPENSN_FUSED_PERSIST: largest level work (cells x angle-groups) for one launch per run.
  std::size_t max_level_work = 0;
  /// OPENSN_FUSED_PBLOCK: threads per block of the one-launch run kernel.
  unsigned int block_size = 256;
};

const PersistKnobs&
GetPersistKnobs()
{
  static const PersistKnobs knobs = []
  {
    PersistKnobs k;
    if (const char* v = std::getenv("OPENSN_FUSED_PERSIST"))
      k.max_level_work = std::strtoul(v, nullptr, 10);
    if (const char* v = std::getenv("OPENSN_FUSED_PBLOCK"))
      k.block_size = static_cast<unsigned int>(std::strtoul(v, nullptr, 10));
    return k;
  }();
  return knobs;
}

} // namespace

std::size_t
AAHDSweepChunk::SubmitBatchRun(std::size_t lane,
                               const std::vector<AAHD_AngleSet*>& batch,
                               bool early_completion)
{
  if (batch.empty())
    return 0;
  FusedLane& run_lane = lanes_[lane];

  // Build the shared (angle-set-independent) arguments from any one representative angle set.
  // The kernel replaces its per-angle-set fields with each thread's batch-slot values.
  auto* front_fluds = dynamic_cast<AAHD_FLUDS*>(&batch.front()->GetFLUDS());
  gpu_kernel::Arguments<SweepKind::AAH> shared_args(
    problem_, groupset_, *batch.front(), *front_fluds, surface_source_active_);

  // The run covers the deepest member's level count, and the angle-group axis covers the
  // largest member. Members with fewer levels or angles get empty slots or idle threads. The
  // kernel checks both bounds against each slot.
  const std::size_t batch_size = batch.size();
  std::size_t max_depth = 0;
  std::size_t max_angleset_size = 0;
  for (auto* as : batch)
  {
    max_depth = std::max(max_depth, as->NumLevels());
    max_angleset_size = std::max(max_angleset_size, as->GetNumAngles());
  }
  // Enqueue the completion copies of the members whose last level is `level` (-1 for members
  // with no levels), then a callback that marks them complete.
  auto complete_members = [&](std::ptrdiff_t level)
  {
    std::vector<std::atomic<bool>*> flags;
    for (auto* as : batch)
      if (static_cast<std::ptrdiff_t>(as->NumLevels()) - 1 == level)
      {
        as->SubmitCompletionCopies();
        flags.push_back(&as->CompletionReady());
      }
    if (not flags.empty())
      run_lane.stream.add_callback(
        [](std::vector<std::atomic<bool>*>&& done)
        {
          for (auto* flag : done)
            flag->store(true, std::memory_order_release);
        },
        std::move(flags));
  };
  if (early_completion)
    complete_members(-1);
  if (max_depth == 0)
  {
    if (not early_completion)
      for (auto* as : batch)
        as->SubmitCompletionCopies();
    return 0;
  }
  const auto stride_size = static_cast<unsigned int>(max_angleset_size * groupset_.GetNumGroups());

  // Precompute every level's batch-slot data on the host. Then copy the whole run to the device
  // at once on the lane's stream. The levels below then launch back to back with no copy or
  // synchronization between them. The lane is idle, so its buffers can be rewritten.
  const std::size_t total_slots = batch_size * max_depth;
  auto& host_slots = run_lane.host_slots;
  if (host_slots.size() < total_slots)
    host_slots.resize(total_slots);
  for (std::size_t m = 0; m < batch_size; ++m)
    batch[m]->FillBatchSlots(max_depth, batch_size, host_slots.data() + m);
  std::vector<unsigned int> max_level_size_per_level(max_depth, 0);
  for (std::size_t level = 0; level < max_depth; ++level)
    for (std::size_t m = 0; m < batch_size; ++m)
      max_level_size_per_level[level] =
        std::max(max_level_size_per_level[level], host_slots[level * batch_size + m].num_cells);
  if (run_lane.device_slots.size() < total_slots)
    run_lane.device_slots = crb::DeviceMemory<gpu_kernel::AAHD_BatchSlot>(total_slots);
  crb::copy(run_lane.device_slots, host_slots, total_slots, 0, 0, run_lane.stream);

  // Compile for the cell DOF count when every local cell has the same one.
  const std::uint32_t fixed_ndofs =
    (min_num_cell_dofs_ == max_num_cell_dofs_) ? max_num_cell_dofs_ : 0;

  auto* device_slots_base = run_lane.device_slots.get();

  // Small runs sweep every level in one launch, with one block per member.
  const auto& persist = GetPersistKnobs();
  const std::size_t max_level_work =
    static_cast<std::size_t>(
      *std::max_element(max_level_size_per_level.begin(), max_level_size_per_level.end())) *
    stride_size;
  if (persist.max_level_work > 0 and max_level_work <= persist.max_level_work)
  {
    auto launch = [&](auto fixed_c)
    {
      LaunchFusedRun<decltype(fixed_c)::value>(run_lane.stream,
                                               shared_args,
                                               device_slots_base,
                                               static_cast<unsigned int>(batch_size),
                                               static_cast<unsigned int>(max_depth),
                                               persist.block_size);
    };
    switch (fixed_ndofs)
    {
      case 2:
        launch(std::integral_constant<std::uint32_t, 2>{});
        break;
      case 3:
        launch(std::integral_constant<std::uint32_t, 3>{});
        break;
      case 4:
        launch(std::integral_constant<std::uint32_t, 4>{});
        break;
      case 6:
        launch(std::integral_constant<std::uint32_t, 6>{});
        break;
      case 8:
        launch(std::integral_constant<std::uint32_t, 8>{});
        break;
      default:
        launch(std::integral_constant<std::uint32_t, 0>{});
        break;
    }
    if (early_completion)
      for (std::size_t level = 0; level < max_depth; ++level)
        complete_members(static_cast<std::ptrdiff_t>(level));
    else
      for (auto* as : batch)
        as->SubmitCompletionCopies();
    return 1;
  }

  for (std::size_t level = 0; level < max_depth; ++level)
  {
    LaunchLevelSlots(run_lane.stream,
                     shared_args,
                     device_slots_base + level * batch_size,
                     batch_size,
                     max_level_size_per_level[level],
                     stride_size,
                     fixed_ndofs,
                     fused_launch_.linear_block_size,
                     fused_launch_.min_small_level_stride,
                     fused_launch_.linear_mapping_min_threads);
    if (early_completion)
      complete_members(static_cast<std::ptrdiff_t>(level));
  }
  // The members' device-to-host copies follow the last level on the same stream. The caller
  // polls IsLaneIdle instead of waiting.
  if (not early_completion)
    for (auto* as : batch)
      as->SubmitCompletionCopies();
  return max_depth;
}

AAHDSweepChunk::~AAHDSweepChunk()
{
  for (auto& step : rolling_)
  {
#if defined(__NVCC__)
    if (step.event)
      cudaEventDestroy(static_cast<cudaEvent_t>(step.event));
#elif defined(__HIPCC__)
    if (step.event)
      (void)hipEventDestroy(static_cast<hipEvent_t>(step.event));
#endif
  }
}

void
AAHDSweepChunk::InitRolling(std::size_t num_streams, std::size_t num_steps, std::size_t capacity)
{
  if (rolling_steps_ == num_steps and rolling_.size() == num_streams * num_steps and
      not rolling_.empty() and rolling_.front().host_slots.size() >= capacity)
    return;
  for (auto& step : rolling_)
  {
#if defined(__NVCC__)
    if (step.event)
      cudaEventDestroy(static_cast<cudaEvent_t>(step.event));
#elif defined(__HIPCC__)
    if (step.event)
      (void)hipEventDestroy(static_cast<hipEvent_t>(step.event));
#endif
  }
  rolling_.clear();
  rolling_steps_ = num_steps;
  rolling_.resize(num_streams * num_steps);
  for (auto& step : rolling_)
  {
    step.host_slots.resize(std::max<std::size_t>(capacity, 1));
    step.device_slots = crb::DeviceMemory<gpu_kernel::AAHD_BatchSlot>(step.host_slots.size());
#if defined(__NVCC__)
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    step.event = event;
#elif defined(__HIPCC__)
    hipEvent_t event;
    (void)hipEventCreateWithFlags(&event, hipEventDisableTiming);
    step.event = event;
#endif
  }
}

bool
AAHDSweepChunk::IsRollingStepDone(std::size_t stream_index, std::size_t ring_index)
{
  auto& step = rolling_[stream_index * rolling_steps_ + ring_index];
#if defined(__NVCC__)
  return cudaEventQuery(static_cast<cudaEvent_t>(step.event)) == cudaSuccess;
#elif defined(__HIPCC__)
  return hipEventQuery(static_cast<hipEvent_t>(step.event)) == hipSuccess;
#else
  return lanes_[stream_index].stream.is_completed();
#endif
}

std::size_t
AAHDSweepChunk::SubmitRollingStep(std::size_t stream_index,
                                  std::size_t ring_index,
                                  std::vector<RollingMember>& active,
                                  std::vector<AAHD_AngleSet*>& completing)
{
  completing.clear();
  RollingStep& step = rolling_[stream_index * rolling_steps_ + ring_index];
  crb::Stream& stream = lanes_[stream_index].stream;

  // One slot per active angle set at its current level. The slots' work is flattened, so each
  // slot records where its (cell, angle-group) pairs start.
  std::size_t count = 0;
  std::size_t total_work = 0;
  AAHD_AngleSet* front = nullptr;
  const std::size_t num_groups = groupset_.GetNumGroups();
  for (auto& m : active)
  {
    if (m.level >= m.angle_set->NumLevels())
      continue;
    m.angle_set->UpdateSlotLevel(m.level, m.slot);
    m.slot.work_offset = static_cast<std::uint32_t>(total_work);
    total_work += static_cast<std::size_t>(m.slot.num_cells) * m.slot.angleset_size * num_groups;
    step.host_slots[count++] = m.slot;
    if (front == nullptr)
      front = m.angle_set;
  }

  std::size_t launches = 0;
  if (count > 0 and total_work > 0)
  {
    crb::copy(step.device_slots, step.host_slots, count, 0, 0, stream);
    auto* front_fluds = dynamic_cast<AAHD_FLUDS*>(&front->GetFLUDS());
    gpu_kernel::Arguments<SweepKind::AAH> shared_args(
      problem_, groupset_, *front, *front_fluds, surface_source_active_);
    const std::uint32_t fixed_ndofs =
      (min_num_cell_dofs_ == max_num_cell_dofs_) ? max_num_cell_dofs_ : 0;
    LaunchFlatLevel(stream,
                    shared_args,
                    step.device_slots.get(),
                    count,
                    total_work,
                    fixed_ndofs,
                    fused_launch_.linear_block_size);
    launches = 1;
  }

  // Angle sets past their last level are complete. The others advance.
  auto keep = active.begin();
  for (auto& m : active)
  {
    const std::size_t num_levels = m.angle_set->NumLevels();
    if (m.level + 1 >= num_levels)
    {
      m.angle_set->SubmitCompletionCopies();
      completing.push_back(m.angle_set);
    }
    else
    {
      ++m.level;
      *keep++ = m;
    }
  }
  active.erase(keep, active.end());

#if defined(__NVCC__)
  cudaEventRecord(static_cast<cudaEvent_t>(step.event), stream);
#elif defined(__HIPCC__)
  (void)hipEventRecord(static_cast<hipEvent_t>(step.event), stream);
#endif
  return launches;
}

} // namespace opensn
