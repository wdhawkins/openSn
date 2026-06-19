// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/boundary_carrier.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/boundary_bank.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/utils/error.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace opensn
{

namespace
{

constexpr unsigned int BoundaryReductionBlockSize = 256;

__CRB_GLOBAL_FUNC__ void
ReduceMaxKernel(const double* input, std::size_t size, double* output)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  __shared__ double shared_max[BoundaryReductionBlockSize];
  const unsigned int tid = threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  __attribute__((opencl_local)) double shared_max[BoundaryReductionBlockSize];
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
  for (unsigned int offset = BoundaryReductionBlockSize / 2; offset > 0; offset /= 2)
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
  for (unsigned int offset = BoundaryReductionBlockSize / 2; offset > 0; offset /= 2)
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
ReduceDeviceMaxToHost(crb::DeviceMemory<double>& device_values)
{
  std::size_t current_size = device_values.size();
  if (current_size == 0)
    return 0.0;

  while (current_size > 1)
  {
    const unsigned int num_blocks =
      static_cast<unsigned int>(std::min<std::size_t>((current_size + BoundaryReductionBlockSize -
                                                       1) /
                                                        BoundaryReductionBlockSize,
                                                      65535));
    crb::DeviceMemory<double> reduced(num_blocks);
#if defined(__NVCC__) || defined(__HIPCC__)
    ReduceMaxKernel<<<num_blocks, BoundaryReductionBlockSize>>>(
      device_values.get(), current_size, reduced.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    crb::Stream stream;
    stream.parallel_for(sycl::nd_range<3>(crb::Dim3(num_blocks) *
                                            crb::Dim3(BoundaryReductionBlockSize),
                                          crb::Dim3(BoundaryReductionBlockSize)),
                        [=](sycl::nd_item<3>)
                        { ReduceMaxKernel(device_values.get(), current_size, reduced.get()); });
    stream.synchronize();
#endif
    device_values = std::move(reduced);
    current_size = num_blocks;
  }

  crb::HostVector<double> host_result(1, 0.0);
  crb::copy(host_result, device_values, 1);
  return host_result[0];
}

__CRB_GLOBAL_FUNC__ void
ComputeBoundaryFluxPointwiseChangeKernel(const double* current,
                                         const double* old,
                                         std::size_t size,
                                         double* block_max)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  __shared__ double shared_max[BoundaryReductionBlockSize];
  const unsigned int tid = threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const std::size_t stride = work_index.get_local_range(2) * work_index.get_group_range(2);
  std::size_t i = work_index.get_global_id(2);
#endif

  double local_max = 0.0;
  while (i < size)
  {
    const double old_value = old[i];
    const double new_value = current[i];
    const double delta = std::fabs(new_value - old_value);
    const double max_value = std::max(std::fabs(new_value), std::fabs(old_value));
    const double change =
      max_value >= std::numeric_limits<double>::min() ? delta / max_value : delta;
    local_max = std::max(local_max, change);
    i += stride;
  }

#if defined(__NVCC__) || defined(__HIPCC__)
  shared_max[tid] = local_max;
  __syncthreads();
  for (unsigned int offset = BoundaryReductionBlockSize / 2; offset > 0; offset /= 2)
  {
    if (tid < offset)
      shared_max[tid] = std::max(shared_max[tid], shared_max[tid + offset]);
    __syncthreads();
  }
  if (tid == 0)
    block_max[blockIdx.x] = shared_max[0];
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  block_max[work_index.get_global_id(2)] = local_max;
#endif
}

} // namespace

BoundaryCarrier::BoundaryCarrier(
  BoundaryBank& bank,
  const std::vector<LBSGroupset>& groupsets,
  const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries)
  : bank_(bank)
{
  if (not bank.IsAllocationDisabled())
    throw std::runtime_error("Cannot create a BoundaryCarrier while BoundaryBank allocation is "
                             "enabled.");

  device_boundary_flux_.reserve(groupsets.size());
  device_boundary_flux_.resize(groupsets.size());
  delayed_ranges_.resize(groupsets.size());

  for (const auto& groupset : groupsets)
  {
    auto& host_data = bank[groupset.id].boundary_flux;
    device_boundary_flux_[groupset.id] = crb::DeviceMemory<double>(host_data.size());
    for (const auto& [_, boundary] : boundaries)
      boundary->GetDelayedAngularFluxStorageRanges(groupset.id, delayed_ranges_[groupset.id]);
  }
}

void
BoundaryCarrier::UploadToDevice(int groupset_id)
{
  auto& host_data = bank_[groupset_id].boundary_flux;
  auto& device_data = device_boundary_flux_[groupset_id];
  crb::copy(device_data, host_data, host_data.size());
}

void
BoundaryCarrier::DownloadToHost(int groupset_id)
{
  auto& host_data = bank_[groupset_id].boundary_flux;
  auto& device_data = device_boundary_flux_[groupset_id];
  crb::copy(host_data, device_data, host_data.size());
}

void
BoundaryCarrier::CopyDelayedAngularFluxNewToOldOnDevice(int groupset_id)
{
  auto& device_data = device_boundary_flux_[groupset_id];
  for (const auto& range : delayed_ranges_[groupset_id])
  {
    if (range.size == 0)
      continue;
    OpenSnLogicalErrorIf(range.current_offset + range.size > device_data.size() or
                           range.old_offset + range.size > device_data.size(),
                         "Boundary delayed angular flux range exceeds device storage.");
    crb::copy(device_data, device_data, range.size, range.current_offset, range.old_offset);
  }
}

double
BoundaryCarrier::ComputeDelayedAngularFluxPointwiseChangeOnDevice(int groupset_id)
{
  const auto& ranges = delayed_ranges_[groupset_id];
  if (ranges.empty())
    return 0.0;

  auto& device_data = device_boundary_flux_[groupset_id];
  double local_max = 0.0;

  for (const auto& range : ranges)
  {
    if (range.size == 0)
      continue;
    OpenSnLogicalErrorIf(range.current_offset + range.size > device_data.size() or
                           range.old_offset + range.size > device_data.size(),
                         "Boundary delayed angular flux range exceeds device storage.");
    const unsigned int num_blocks =
      static_cast<unsigned int>(std::min<std::size_t>((range.size + BoundaryReductionBlockSize - 1) /
                                                        BoundaryReductionBlockSize,
                                                      65535));
    crb::DeviceMemory<double> device_block_max(num_blocks);
#if defined(__NVCC__) || defined(__HIPCC__)
    ComputeBoundaryFluxPointwiseChangeKernel<<<num_blocks, BoundaryReductionBlockSize>>>(
      device_data.get() + range.current_offset,
      device_data.get() + range.old_offset,
      range.size,
      device_block_max.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    crb::Stream stream;
    crb::Dim3 block_size(BoundaryReductionBlockSize);
    crb::Dim3 grid_size(num_blocks);
    stream.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<3>)
                        {
                          ComputeBoundaryFluxPointwiseChangeKernel(
                            device_data.get() + range.current_offset,
                            device_data.get() + range.old_offset,
                            range.size,
                            device_block_max.get());
                        });
#endif
    local_max = std::max(local_max, ReduceDeviceMaxToHost(device_block_max));
  }

  return local_max;
}

} // namespace opensn
