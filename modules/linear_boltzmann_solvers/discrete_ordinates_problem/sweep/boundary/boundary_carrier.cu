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

constexpr unsigned int kBoundaryReductionBlockSize = 256;

__CRB_GLOBAL_FUNC__ void
ComputeBoundaryFluxPointwiseChangeKernel(const double* current,
                                         const double* old,
                                         std::size_t size,
                                         double* block_max)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  __shared__ double shared_max[kBoundaryReductionBlockSize];
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
  for (unsigned int offset = kBoundaryReductionBlockSize / 2; offset > 0; offset /= 2)
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
      static_cast<unsigned int>(std::min<std::size_t>((range.size + kBoundaryReductionBlockSize - 1) /
                                                        kBoundaryReductionBlockSize,
                                                      65535));
    crb::DeviceMemory<double> device_block_max(num_blocks);
    crb::HostVector<double> host_block_max(num_blocks, 0.0);
#if defined(__NVCC__) || defined(__HIPCC__)
    ComputeBoundaryFluxPointwiseChangeKernel<<<num_blocks, kBoundaryReductionBlockSize>>>(
      device_data.get() + range.current_offset,
      device_data.get() + range.old_offset,
      range.size,
      device_block_max.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    crb::Stream stream;
    crb::Dim3 block_size(kBoundaryReductionBlockSize);
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
    crb::copy(host_block_max, device_block_max, host_block_max.size());
    for (const double value : host_block_max)
      local_max = std::max(local_max, value);
  }

  return local_max;
}

} // namespace opensn
