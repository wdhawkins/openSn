// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/device_classic_richardson_runtime.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/sweep_wgs_context.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/quadrature_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/source_xs_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/mesh_view.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/quadrature_view.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/outflow/outflow_carrier.h"
#include "framework/runtime.h"
#include "framework/utils/caliper_scopes.h"
#include "framework/utils/error.h"
#include "framework/utils/timer.h"
#include "caribou/main.hpp"
#include <atomic>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace crb = caribou;

namespace opensn
{

namespace
{

constexpr unsigned int kBlockSize = 256;
crb::DeviceMemory<double> phi_change_block_max_scratch;
crb::DeviceMemory<double> phi_change_reduce_scratch;

__CRB_GLOBAL_FUNC__ void
ReduceMaxKernel(const double* input, std::size_t size, double* output)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  __shared__ double shared_max[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  __attribute__((opencl_local)) double shared_max[kBlockSize];
  const unsigned int tid = static_cast<unsigned int>(work_index.get_local_id(2));
  const std::size_t stride = work_index.get_local_range(2) * work_index.get_group_range(2);
  std::size_t i = work_index.get_global_id(2);
#endif

  double local_max = 0.0;
  while (i < size)
  {
    local_max = std::max(local_max, input[i]);
    i += stride;
  }

#if defined(__NVCC__) || defined(__HIPCC__)
  shared_max[tid] = local_max;
  __syncthreads();
  for (unsigned int offset = kBlockSize / 2; offset > 0; offset /= 2)
  {
    if (tid < offset)
      shared_max[tid] = std::max(shared_max[tid], shared_max[tid + offset]);
    __syncthreads();
  }
  if (tid == 0)
    output[blockIdx.x] = shared_max[0];
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  shared_max[tid] = local_max;
  work_index.barrier(sycl::access::fence_space::local_space);
  for (unsigned int offset = kBlockSize / 2; offset > 0; offset /= 2)
  {
    if (tid < offset)
      shared_max[tid] = std::max(shared_max[tid], shared_max[tid + offset]);
    work_index.barrier(sycl::access::fence_space::local_space);
  }
  if (tid == 0)
    output[work_index.get_group(2)] = shared_max[0];
#endif
}

double
ReduceDeviceMaxToHost(const double* device_values,
                      std::size_t size,
                      crb::DeviceMemory<double>& reduce_scratch)
{
  std::size_t current_size = size;
  if (current_size == 0)
    return 0.0;

  const auto required_scratch = std::max<std::size_t>(1, current_size);
  if (reduce_scratch.size() < required_scratch)
    reduce_scratch = crb::DeviceMemory<double>(required_scratch);

  const double* current_values = device_values;

  while (current_size > 1)
  {
    const auto num_blocks =
      static_cast<unsigned int>(std::min<std::size_t>((current_size + kBlockSize - 1) / kBlockSize,
                                                      65535));
#if defined(__NVCC__) || defined(__HIPCC__)
    ReduceMaxKernel<<<num_blocks, kBlockSize>>>(current_values, current_size, reduce_scratch.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    crb::Stream stream;
    stream.parallel_for(sycl::nd_range<3>(crb::Dim3(num_blocks) * crb::Dim3(kBlockSize),
                                          crb::Dim3(kBlockSize)),
                        [=](sycl::nd_item<3>)
                        { ReduceMaxKernel(current_values, current_size, reduce_scratch.get()); });
    stream.synchronize();
#endif
    current_values = reduce_scratch.get();
    current_size = num_blocks;
  }

  crb::HostVector<double> host_result(1, 0.0);
  crb::copy(host_result, reduce_scratch, 1);
  return host_result[0];
}

AAHDSweepChunk&
GetAAHDSweepChunk(SweepWGSContext& sweep_context)
{
  auto sweep_chunk = std::dynamic_pointer_cast<AAHDSweepChunk>(sweep_context.sweep_chunk);
  OpenSnLogicalErrorIf(not sweep_chunk,
                       "DeviceClassicRichardsonRuntime requires an AAHD sweep chunk.");
  return *sweep_chunk;
}

__CRB_GLOBAL_FUNC__ void
BuildDeviceClassicRichardsonSourceKernel(const char* mesh_data,
                                         const char* quad_data,
                                         const double* source_base,
                                         double* source,
                                         const double* phi_old,
                                         std::uint32_t num_groups,
                                         std::uint32_t groupset_start,
                                         std::uint32_t groupset_size,
                                         std::uint32_t max_cell_dofs,
                                         bool apply_wgs_scatter,
                                         bool apply_wgs_fission)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  const std::uint32_t cell_idx = blockIdx.y;
  const std::uint32_t local_idx = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const std::uint32_t cell_idx = work_index.get_group(1);
  const std::uint32_t local_idx = work_index.get_global_id(2);
#endif

  MeshView mesh(mesh_data);
  if (cell_idx >= mesh.num_cells)
    return;

  CellView cell;
  mesh.GetCellView(cell, cell_idx);
  QuadratureView quadrature(quad_data);

  const std::uint32_t num_moments = quadrature.num_moments;
  const std::uint32_t work_per_cell = max_cell_dofs * num_moments * groupset_size;
  if (local_idx >= work_per_cell)
    return;

  const std::uint32_t group_idx = local_idx % groupset_size;
  const std::uint32_t moment_idx = (local_idx / groupset_size) % num_moments;
  const std::uint32_t node_idx = local_idx / (groupset_size * num_moments);
  if (node_idx >= cell.num_nodes)
    return;

  const std::uint32_t g = groupset_start + group_idx;
  const std::uint64_t dof = cell.phi_address +
                            (static_cast<std::uint64_t>(node_idx) * num_moments + moment_idx) *
                              num_groups +
                            g;

  double rhs = source_base[dof];
  const auto* xs = cell.source_xs;
  const std::uint64_t phi_moment_base =
    cell.phi_address + (static_cast<std::uint64_t>(node_idx) * num_moments + moment_idx) *
                         num_groups;

  if (apply_wgs_scatter and xs->has_scatter)
  {
    const std::uint32_t ell = quadrature.moment_ell[moment_idx];
    if (ell < xs->num_moments)
    {
      const std::uint64_t xs_num_groups = xs->num_groups;
      const double* row = xs->scatter +
                          (static_cast<std::uint64_t>(ell) * xs_num_groups + g) * xs_num_groups;
      for (std::uint32_t gp = groupset_start; gp < groupset_start + groupset_size; ++gp)
        rhs += row[gp] * phi_old[phi_moment_base + gp];
    }
  }

  if (apply_wgs_fission and xs->is_fissionable and moment_idx == 0)
  {
    const double* row = xs->production + static_cast<std::uint64_t>(g) * xs->num_groups;
    const std::uint64_t phi_base =
      cell.phi_address + static_cast<std::uint64_t>(node_idx) * num_moments * num_groups;
    for (std::uint32_t gp = groupset_start; gp < groupset_start + groupset_size; ++gp)
      rhs += row[gp] * phi_old[phi_base + gp];
  }

  source[dof] = rhs;
}

__CRB_GLOBAL_FUNC__ void
ZeroDeviceGroupsetPhiKernel(double* phi,
                            std::size_t size,
                            std::uint32_t num_groups,
                            std::uint32_t groupset_start,
                            std::uint32_t groupset_size)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  std::size_t i = work_index.get_global_id(2);
  const std::size_t stride = work_index.get_local_range(2) * work_index.get_group_range(2);
#endif

  while (i < size)
  {
    const std::uint32_t g = i % num_groups;
    if (g >= groupset_start and g < groupset_start + groupset_size)
      phi[i] = 0.0;
    i += stride;
  }
}

__CRB_GLOBAL_FUNC__ void
CopyDeviceGroupsetPhiKernel(double* phi_old,
                            const double* phi_new,
                            std::size_t size,
                            std::uint32_t num_groups,
                            std::uint32_t groupset_start,
                            std::uint32_t groupset_size)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  std::size_t i = work_index.get_global_id(2);
  const std::size_t stride = work_index.get_local_range(2) * work_index.get_group_range(2);
#endif

  while (i < size)
  {
    const std::uint32_t g = i % num_groups;
    if (g >= groupset_start and g < groupset_start + groupset_size)
      phi_old[i] = phi_new[i];
    i += stride;
  }
}

__CRB_GLOBAL_FUNC__ void
ComputePhiChangeKernel(const double* phi_new,
                       const double* phi_old,
                       std::size_t size,
                       std::uint32_t num_groups,
                       std::uint32_t groupset_start,
                       std::uint32_t groupset_size,
                       double* block_results)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  __shared__ double block_change[kBlockSize];
  const unsigned int tid = threadIdx.x;
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const std::size_t i0 = work_index.get_global_id(2);
  std::size_t i = i0;
  const std::size_t stride = work_index.get_local_range(2) * work_index.get_group_range(2);
  double local_change = 0.0;
#endif

#if defined(__NVCC__) || defined(__HIPCC__)
  double local_change = 0.0;
#endif
  while (i < size)
  {
    const std::uint32_t g = i % num_groups;
    if (g >= groupset_start and g < groupset_start + groupset_size)
    {
      const double old_value = phi_old[i];
      const double new_value = phi_new[i];
      const double delta = std::fabs(new_value - old_value);
      const double scale = std::max(std::fabs(new_value), std::fabs(old_value));
      const double change =
        scale >= std::numeric_limits<double>::min() ? delta / scale : delta;
      local_change = std::max(local_change, change);
    }
    i += stride;
  }

#if defined(__NVCC__) || defined(__HIPCC__)
  block_change[tid] = local_change;
  __syncthreads();
  for (unsigned int offset = kBlockSize / 2; offset > 0; offset /= 2)
  {
    if (tid < offset)
      block_change[tid] = std::max(block_change[tid], block_change[tid + offset]);
    __syncthreads();
  }
  if (tid == 0)
    block_results[blockIdx.x] = block_change[0];
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  block_results[i0] = local_change;
#endif
}

} // namespace

using namespace std::chrono;

DeviceClassicRichardsonRuntime::DeviceClassicRichardsonRuntime(SweepWGSContext& sweep_context)
  : sweep_context_(&sweep_context),
    problem_(&sweep_context.do_problem),
    groupset_(&sweep_context.groupset),
    sweep_chunk_(&GetAAHDSweepChunk(sweep_context))
{
  OpenSnLogicalErrorIf(not problem_->UseGPUs(),
                       "DeviceClassicRichardsonRuntime requires GPU support.");
  OpenSnLogicalErrorIf(problem_->GetSweepType() != "AAH",
                       "DeviceClassicRichardsonRuntime requires AAH sweeps.");
  OpenSnLogicalErrorIf(sweep_context.sweep_chunk == nullptr,
                       "DeviceClassicRichardsonRuntime requires a sweep chunk.");

  angle_sets_.reserve(groupset_->angle_agg->GetNumAngleSets());
  for (const auto& angle_set : groupset_->angle_agg->GetAngleSetGroups())
  {
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_set.get());
    OpenSnLogicalErrorIf(aahd_angle_set == nullptr,
                         "DeviceClassicRichardsonRuntime requires AAHD angle sets.");
    angle_sets_.push_back(aahd_angle_set);
  }

  InitializeSweepResources();
}

void
DeviceClassicRichardsonRuntime::CopyPhiOldToDevice()
{
  if (!problem_->UseGPUs())
    return;
  problem_->GetPhiOldPinner()->CopyToDevice();
}

void
DeviceClassicRichardsonRuntime::CopyPhiOldToHost()
{
  if (!problem_->UseGPUs())
    return;
  problem_->GetPhiOldPinner()->CopyFromDevice();
}

void
DeviceClassicRichardsonRuntime::CopySourceMomentsToDevice()
{
  if (!problem_->UseGPUs())
    return;
  problem_->GetSourceMomentsPinner()->CopyToDevice();
}

void
DeviceClassicRichardsonRuntime::CopyDevicePhiNewToOld(const LBSGroupset& groupset)
{
  if (!problem_->UseGPUs())
    return;

  const auto phi_size = problem_->GetPhiNewLocal().size();
  const std::uint32_t block_size = kBlockSize;
  const std::uint32_t grid_size =
    static_cast<std::uint32_t>((phi_size + block_size - 1) / block_size);

#if defined(__NVCC__) || defined(__HIPCC__)
  CopyDeviceGroupsetPhiKernel<<<grid_size, block_size>>>(problem_->GetPhiOldPinner()->GetDevicePtr(),
                                                         problem_->GetPhiPinner()->GetDevicePtr(),
                                                         phi_size,
                                                         problem_->GetNumGroups(),
                                                         groupset.first_group,
                                                         groupset.GetNumGroups());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  crb::Stream stream;
  stream.parallel_for(sycl::nd_range<3>(crb::Dim3(grid_size) * crb::Dim3(block_size),
                                        crb::Dim3(block_size)),
                      [=](sycl::nd_item<3>)
                      {
                        CopyDeviceGroupsetPhiKernel(problem_->GetPhiOldPinner()->GetDevicePtr(),
                                                   problem_->GetPhiPinner()->GetDevicePtr(),
                                                   phi_size,
                                                   problem_->GetNumGroups(),
                                                   groupset.first_group,
                                                   groupset.GetNumGroups());
                      });
  stream.synchronize();
#endif
}

void
DeviceClassicRichardsonRuntime::CopySourceBaseToDevice()
{
  if (!problem_->UseGPUs())
    return;
  problem_->GetSourceBaseMomentsPinner()->CopyToDevice();
}

void
DeviceClassicRichardsonRuntime::InitializeSweepResources()
{
  pool_.Resize(angle_sets_.size());
  execution_order_.reserve(angle_sets_.size());

  int local_max_num_messages = 0;
  for (auto* angle_set : angle_sets_)
  {
    angle_set->InitializeDelayedUpstreamData();
    local_max_num_messages = std::max(angle_set->GetMaxBufferMessages(), local_max_num_messages);
  }

  int global_max_num_messages = 0;
  mpi_comm.all_reduce(local_max_num_messages, global_max_num_messages, mpi::op::max<int>());

  for (auto* angle_set : angle_sets_)
    angle_set->SetMaxBufferMessages(global_max_num_messages);
}

void
DeviceClassicRichardsonRuntime::PrepareSweep(bool use_boundary_source,
                                             bool zero_incoming_delayed_psi)
{
  if (zero_incoming_delayed_psi)
  {
    for (const auto& [boundary_id, boundary] : problem_->GetSweepBoundaries())
      boundary->ZeroOpposingDelayedAngularFluxOld(groupset_->id);

    for (auto* angle_set : angle_sets_)
    {
      auto& fluds = angle_set->GetFLUDS();
      std::fill(fluds.DelayedLocalPsiOld().begin(), fluds.DelayedLocalPsiOld().end(), 0.0);
      for (auto& loc_vector : fluds.DelayedPrelocIOutgoingPsiOld())
        std::fill(loc_vector.begin(), loc_vector.end(), 0.0);
    }
  }

  sweep_chunk_->ZeroDestinationPsi();
  sweep_chunk_->ZeroDestinationPhi();
  sweep_chunk_->SetBoundarySourceActiveFlag(use_boundary_source);

  const auto phi_size = problem_->GetPhiNewLocal().size();
  const std::uint32_t block_size = kBlockSize;
  const std::uint32_t grid_size =
    static_cast<std::uint32_t>((phi_size + block_size - 1) / block_size);

#if defined(__NVCC__) || defined(__HIPCC__)
  ZeroDeviceGroupsetPhiKernel<<<grid_size, block_size>>>(problem_->GetPhiPinner()->GetDevicePtr(),
                                                         phi_size,
                                                         problem_->GetNumGroups(),
                                                         groupset_->first_group,
                                                         groupset_->GetNumGroups());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  crb::Stream stream;
  stream.parallel_for(sycl::nd_range<3>(crb::Dim3(grid_size) * crb::Dim3(block_size),
                                        crb::Dim3(block_size)),
                      [=](sycl::nd_item<3>)
                      {
                        ZeroDeviceGroupsetPhiKernel(problem_->GetPhiPinner()->GetDevicePtr(),
                                                   phi_size,
                                                   problem_->GetNumGroups(),
                                                   groupset_->first_group,
                                                   groupset_->GetNumGroups());
                      });
  stream.synchronize();
#endif
  problem_->GetOutflowCarrier()->ZeroGroupsOnDevice(groupset_->first_group,
                                                    groupset_->GetNumGroups(),
                                                    problem_->GetNumGroups());
}

void
DeviceClassicRichardsonRuntime::BuildSource(const LBSGroupset& groupset,
                                            bool apply_wgs_scatter,
                                            bool apply_wgs_fission)
{
  if (!problem_->UseGPUs())
    return;

  auto* mesh = problem_->GetMeshCarrier();
  auto* quadrature = groupset.quad_carrier.get();
  const std::uint32_t max_cell_dofs = problem_->GetMaxCellDOFCount();
  const std::uint32_t work_per_cell =
    max_cell_dofs * problem_->GetNumMoments() * groupset.GetNumGroups();
  const std::uint32_t block_size = kBlockSize;
  const std::uint32_t grid_size_x = (work_per_cell + block_size - 1) / block_size;
  crb::Dim3 block(block_size);
  crb::Dim3 grid(grid_size_x, problem_->GetGrid()->local_cells.size());

#if defined(__NVCC__) || defined(__HIPCC__)
  BuildDeviceClassicRichardsonSourceKernel<<<grid, block>>>(mesh->GetDevicePtr(),
                                                            quadrature->GetDevicePtr(),
                                                            problem_->GetSourceBaseMomentsPinner()
                                                              ->GetDevicePtr(),
                                                            problem_->GetSourceMomentsPinner()
                                                              ->GetDevicePtr(),
                                                            problem_->GetPhiOldPinner()
                                                              ->GetDevicePtr(),
                                                            problem_->GetNumGroups(),
                                                            groupset.first_group,
                                                            groupset.GetNumGroups(),
                                                            max_cell_dofs,
                                                            apply_wgs_scatter,
                                                            apply_wgs_fission);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  crb::Stream stream;
  stream.parallel_for(sycl::nd_range<3>(grid * block, block),
                      [=](sycl::nd_item<3>)
                      {
                        BuildDeviceClassicRichardsonSourceKernel(
                          mesh->GetDevicePtr(),
                          quadrature->GetDevicePtr(),
                          problem_->GetSourceBaseMomentsPinner()->GetDevicePtr(),
                          problem_->GetSourceMomentsPinner()->GetDevicePtr(),
                          problem_->GetPhiOldPinner()->GetDevicePtr(),
                          problem_->GetNumGroups(),
                          groupset.first_group,
                          groupset.GetNumGroups(),
                          max_cell_dofs,
                          apply_wgs_scatter,
                          apply_wgs_fission);
                      });
  stream.synchronize();
#endif
}

double
DeviceClassicRichardsonRuntime::ComputeLocalPhiChange(const LBSGroupset& groupset) const
{
  if (!problem_->UseGPUs())
    return 0.0;

  const std::size_t size = problem_->GetPhiNewLocal().size();
  const std::uint32_t num_blocks =
    static_cast<std::uint32_t>(std::min<std::size_t>((size + kBlockSize - 1) / kBlockSize, 65535));
  if (phi_change_block_max_scratch.size() != num_blocks)
    phi_change_block_max_scratch = crb::DeviceMemory<double>(num_blocks);
#if defined(__NVCC__) || defined(__HIPCC__)
  ComputePhiChangeKernel<<<num_blocks, kBlockSize>>>(problem_->GetPhiPinner()->GetDevicePtr(),
                                                     problem_->GetPhiOldPinner()->GetDevicePtr(),
                                                     size,
                                                     problem_->GetNumGroups(),
                                                     groupset.first_group,
                                                     groupset.GetNumGroups(),
                                                     phi_change_block_max_scratch.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  crb::Stream stream;
  crb::Dim3 block(kBlockSize);
  crb::Dim3 grid(num_blocks);
  stream.parallel_for(sycl::nd_range<3>(grid * block, block),
                      [=](sycl::nd_item<3>)
                      {
                        ComputePhiChangeKernel(problem_->GetPhiPinner()->GetDevicePtr(),
                                               problem_->GetPhiOldPinner()->GetDevicePtr(),
                                               size,
                                               problem_->GetNumGroups(),
                                               groupset.first_group,
                                               groupset.GetNumGroups(),
                                               phi_change_block_max_scratch.get());
                      });
  stream.synchronize();
#endif

  return ReduceDeviceMaxToHost(phi_change_block_max_scratch.get(),
                               num_blocks,
                               phi_change_reduce_scratch);
}

double
DeviceClassicRichardsonRuntime::ComputeGlobalPhiChange(const LBSGroupset& groupset) const
{
  const double local_change = ComputeLocalPhiChange(groupset);
  double global_change = 0.0;
  opensn::mpi_comm.all_reduce(local_change, global_change, mpi::op::max<double>());
  return global_change;
}

DeviceRichardsonConvergenceMetrics
DeviceClassicRichardsonRuntime::ComputeConvergenceMetrics(const LBSGroupset& groupset) const
{
  DeviceRichardsonConvergenceMetrics metrics;
  metrics.phi_change = ComputeLocalPhiChange(groupset);
  metrics.psi_change = last_local_delayed_psi_relative_change_;

  std::array<double, 2> local_metrics = {metrics.phi_change, metrics.psi_change};
  std::array<double, 2> global_metrics = {0.0, 0.0};
  opensn::mpi_comm.all_reduce(
    local_metrics.data(), 2, global_metrics.data(), mpi::op::max<double>());

  metrics.phi_change = global_metrics[0];
  metrics.psi_change = global_metrics[1];
  return metrics;
}

void
DeviceClassicRichardsonRuntime::UploadDelayedPsiToDevice()
{
  for (auto* angle_set : angle_sets_)
  {
    auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS());
    OpenSnLogicalErrorIf(aahd_fluds == nullptr,
                         "DeviceClassicRichardsonRuntime requires AAHD_FLUDS.");
    aahd_fluds->CopyDelayedPsiToDevice();
  }

  for (auto* angle_set : angle_sets_)
  {
    auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS());
    OpenSnLogicalErrorIf(aahd_fluds == nullptr,
                         "DeviceClassicRichardsonRuntime requires AAHD_FLUDS.");
    aahd_fluds->GetStream().synchronize();
  }
}

void
DeviceClassicRichardsonRuntime::DownloadDelayedPsiToHost()
{
  for (auto* angle_set : angle_sets_)
  {
    auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS());
    OpenSnLogicalErrorIf(aahd_fluds == nullptr,
                         "DeviceClassicRichardsonRuntime requires AAHD_FLUDS.");
    aahd_fluds->CopyLocalDelayedPsiFromDevice();
    aahd_fluds->CopyDelayedIncomingPsiCurrentFromDevice();
  }

  for (auto* angle_set : angle_sets_)
  {
    auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS());
    OpenSnLogicalErrorIf(aahd_fluds == nullptr,
                         "DeviceClassicRichardsonRuntime requires AAHD_FLUDS.");
    aahd_fluds->GetStream().synchronize();
  }
}

void
DeviceClassicRichardsonRuntime::ExecuteSweepPass(bool final_download)
{
  last_sweep_pass_count_ = 1;
  last_local_delayed_psi_relative_change_ = 0.0;
  last_delayed_psi_relative_change_ = 0.0;

#if OPENSN_GPU_AWARE_MPI
  constexpr bool use_device_buffers = true;
#else
  constexpr bool use_device_buffers = false;
#endif
  constexpr bool delayed_psi_on_device = true;
  const bool download_delayed_psi = final_download;

  for (auto* angle_set : angle_sets_)
    static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->ZeroDelayedPsiNewOnDevice();

  execution_order_.resize(angle_sets_.size());
  for (std::size_t i = 0; i < angle_sets_.size(); ++i)
  {
    auto* angle_set = angle_sets_[i];
    angle_set->ResetDependencyCounter();
    angle_set->PrepostReceives(use_device_buffers, delayed_psi_on_device);
    execution_order_[i] = i;
  }

  while (not execution_order_.empty())
  {
    std::vector<std::size_t> ready_indices;
    ready_indices.reserve(execution_order_.size());

    for (auto it = execution_order_.begin(); it != execution_order_.end();)
    {
      auto* angle_set = angle_sets_[*it];
      if (angle_set->IsReady())
      {
        ready_indices.push_back(*it);
        std::swap(*it, execution_order_.back());
        execution_order_.pop_back();
      }
      else
        ++it;
    }

    if (ready_indices.empty())
    {
      std::this_thread::yield();
      continue;
    }

    std::atomic_size_t next_ready_idx = 0;
    pool_.ExecuteBatch(
      [this, &ready_indices, &next_ready_idx, final_download, download_delayed_psi](std::size_t)
      {
        while (true)
        {
          const auto ready_pos = next_ready_idx.fetch_add(1, std::memory_order_relaxed);
          if (ready_pos >= ready_indices.size())
            return;

          auto* angle_set = angle_sets_[ready_indices[ready_pos]];
          angle_set->SweepKernelAndSync(*sweep_chunk_, use_device_buffers);
          angle_set->SendAfterFirstPass(use_device_buffers);
          angle_set->FinalizeAfterSweep(
            *sweep_chunk_, use_device_buffers, final_download, download_delayed_psi);
        }
      });
  }

  for (auto* angle_set : angle_sets_)
    angle_set->WaitForDownstreamAndDelayed();

  if (not use_device_buffers)
  {
    for (auto* angle_set : angle_sets_)
      static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->UploadDelayedIncomingPsiCurrentToDevice();
  }

  double local_max_change = 0.0;
  for (auto* angle_set : angle_sets_)
  {
    const double pointwise_change =
      static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->ComputeDelayedPsiPointwiseChangeOnDevice();
    local_max_change = std::max(local_max_change, pointwise_change);
  }
  local_max_change =
    std::max(local_max_change, problem_->ComputeDeviceBoundaryDelayedPsiPointwiseChange(groupset_->id));
  last_local_delayed_psi_relative_change_ = local_max_change;
  last_delayed_psi_relative_change_ = local_max_change;

  for (auto* angle_set : angle_sets_)
    static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->CopyDelayedPsiNewToOldOnDevice();
  problem_->CopyDeviceBoundaryDelayedPsiNewToOld(groupset_->id);
  for (auto* angle_set : angle_sets_)
    angle_set->ResetSweepBuffers();
}

void
DeviceClassicRichardsonRuntime::AccumulateSweepStats(double sweep_time_seconds, size_t sweep_passes)
{
  sweep_context_->sweep_stats_.total_sweep_time += sweep_time_seconds;
  sweep_context_->sweep_stats_.num_sweeps += sweep_passes;
}

void
DeviceClassicRichardsonRuntime::ApplyInverseTransportOperator(SourceFlags scope)
{
  CaliperRegionScope cali_sweep_region("Sweep", CaliperSweepScopeDepth());

  ++sweep_context_->counter_applications_of_inv_op;

  const bool use_boundary_source =
    (scope & APPLY_FIXED_SOURCES) and (not problem_->GetOptions().use_src_moments);
  const bool zero_incoming_delayed_psi = (scope & ZERO_INCOMING_DELAYED_PSI);
  PrepareSweep(use_boundary_source, zero_incoming_delayed_psi);

  const high_resolution_clock::time_point sweep_start = high_resolution_clock::now();
  ExecuteSweepPass(false);
  const high_resolution_clock::time_point sweep_end = high_resolution_clock::now();

  const auto sweep_time =
    static_cast<double>(duration_cast<nanoseconds>(sweep_end - sweep_start).count()) / 1.0e+9;
  AccumulateSweepStats(sweep_time, last_sweep_pass_count_);
}

void
DeviceClassicRichardsonRuntime::FinalizeAngularFluxes()
{
  for (auto* angle_set : angle_sets_)
  {
    auto* fluds = static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS());
    fluds->CopySaveAngularFluxFromDevice();
    fluds->GetStream().synchronize();
    fluds->CopySaveAngularFluxToDestinationPsi(*problem_, *groupset_, *angle_set);
  }
}

} // namespace opensn
