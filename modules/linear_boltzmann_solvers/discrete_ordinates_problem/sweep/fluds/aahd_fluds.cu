// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "framework/utils/error.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace opensn
{
namespace
{

constexpr unsigned int kReductionBlockSize = 256;

__CRB_GLOBAL_FUNC__ void
ComputePsiPointwiseChangeKernel(const double* psi_old,
                                const double* psi_new,
                                std::size_t size,
                                double* block_max)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  __shared__ double shared_max[kReductionBlockSize];
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
    const double old_value = psi_old[i];
    const double new_value = psi_new[i];
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
  for (unsigned int offset = kReductionBlockSize / 2; offset > 0; offset /= 2)
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

void
AccumulatePsiPointwiseChange(const crb::DeviceMemory<double>& old_storage,
                             const crb::DeviceMemory<double>& new_storage,
                             double& local_max,
                             crb::Stream& stream)
{
  const std::size_t size = old_storage.size();
  if (size == 0)
    return;
  OpenSnLogicalErrorIf(size != new_storage.size(),
                       "Delayed angular flux convergence check has mismatched bank sizes.");

  const unsigned int num_blocks =
    static_cast<unsigned int>(std::min<std::size_t>((size + kReductionBlockSize - 1) /
                                                      kReductionBlockSize,
                                                    65535));
  crb::DeviceMemory<double> device_block_max(num_blocks);
  crb::HostVector<double> host_block_max(num_blocks, 0.0);
#if defined(__NVCC__) || defined(__HIPCC__)
  ComputePsiPointwiseChangeKernel<<<num_blocks, kReductionBlockSize, 0, stream>>>(
    old_storage.get(), new_storage.get(), size, device_block_max.get());
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  const crb::Dim3 block_size(kReductionBlockSize);
  const crb::Dim3 grid_size(num_blocks);
  stream.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                      [=](sycl::nd_item<3>)
                      {
                        ComputePsiPointwiseChangeKernel(
                          old_storage.get(), new_storage.get(), size, device_block_max.get());
                      });
#endif
  crb::copy(host_block_max, device_block_max, host_block_max.size(), 0, 0, stream);
  stream.synchronize();
  for (const double value : host_block_max)
    local_max = std::max(local_max, value);
}

} // namespace

AAHD_Bank::AAHD_Bank(const AAHD_Bank& other)
  : host_storage(other.host_storage), device_storage(other.device_storage.size())
{
  crb::copy(device_storage, other.device_storage, device_storage.size());
}

AAHD_Bank&
AAHD_Bank::operator=(const AAHD_Bank& other)
{
  if (this != &other)
  {
    host_storage = other.host_storage;
    if (device_storage.size() != other.device_storage.size())
      device_storage = crb::DeviceMemory<double>(other.device_storage.size());
    crb::copy(device_storage, other.device_storage, device_storage.size());
  }
  return *this;
}

void
AAHD_Bank::UploadToDevice()
{
  crb::copy(device_storage, host_storage, host_storage.size());
}

void
AAHD_Bank::UploadToDevice(crb::Stream& stream)
{
  crb::copy(device_storage, host_storage, host_storage.size(), 0, 0, stream);
}

void
AAHD_Bank::DownloadToHost()
{
  crb::copy(host_storage, device_storage, host_storage.size());
}

void
AAHD_Bank::DownloadToHost(crb::Stream& stream)
{
  crb::copy(host_storage, device_storage, host_storage.size(), 0, 0, stream);
}

void
AAHD_Bank::Clear()
{
  host_storage.clear();
  device_storage.reset();
}

void
AAHD_Bank::CopyDeviceFrom(const AAHD_Bank& other, crb::Stream& stream)
{
  if (device_storage.size() > 0)
    crb::copy(device_storage, other.device_storage, device_storage.size(), 0, 0, stream);
}

void
AAHD_Bank::ZeroDevice()
{
  if (device_storage.size() > 0)
    device_storage.zero_fill();
}

AAHD_NonLocalBank::AAHD_NonLocalBank(const std::vector<std::size_t>& loc_sizes,
                                     const std::vector<std::size_t>& loc_offsets,
                                     std::size_t stride)
  : location_sizes(&loc_sizes), location_offsets(&loc_offsets), stride_size(stride)
{
  host_storage = crb::HostVector<double>(loc_offsets.back() * stride_size, 0.0);
  device_storage = crb::DeviceMemory<double>(loc_offsets.back() * stride_size);
}

void
AAHD_NonLocalBank::UpdateViews(std::vector<std::span<double>>& views)
{
  views.resize(location_sizes->size());
  for (std::size_t i = 0; i < location_sizes->size(); ++i)
  {
    views[i] = std::span<double>(host_storage.data() + (*location_offsets)[i] * stride_size,
                                 (*location_sizes)[i] * stride_size);
  }
}

AAHD_NonLocalDelayedBank::AAHD_NonLocalDelayedBank(const std::vector<std::size_t>& loc_sizes,
                                                   const std::vector<std::size_t>& loc_offsets,
                                                   std::size_t stride)
  : AAHD_NonLocalBank(loc_sizes, loc_offsets, stride)
{
  host_current_storage = crb::HostVector<double>(loc_offsets.back() * stride_size, 0.0);
  device_current_storage = crb::DeviceMemory<double>(loc_offsets.back() * stride_size);
}

void
AAHD_NonLocalDelayedBank::UpdateViews(std::vector<std::span<double>>& current_delayed_views,
                                      std::vector<std::span<double>>& old_delayed_views)
{
  current_delayed_views.resize(location_sizes->size());
  old_delayed_views.resize(location_sizes->size());
  for (std::size_t i = 0; i < location_sizes->size(); ++i)
  {
    current_delayed_views[i] =
      std::span<double>(host_current_storage.data() + (*location_offsets)[i] * stride_size,
                        (*location_sizes)[i] * stride_size);
    old_delayed_views[i] =
      std::span<double>(host_storage.data() + (*location_offsets)[i] * stride_size,
                        (*location_sizes)[i] * stride_size);
  }
}

void
AAHD_NonLocalDelayedBank::SetOldToNew()
{
  std::memcpy(
    host_current_storage.data(), host_storage.data(), host_storage.size() * sizeof(double));
}

void
AAHD_NonLocalDelayedBank::SetNewToOld()
{
  std::memcpy(
    host_storage.data(), host_current_storage.data(), host_storage.size() * sizeof(double));
}

void
AAHD_NonLocalDelayedBank::UploadCurrentToDevice(crb::Stream& stream)
{
  if (host_current_storage.size() > 0)
    crb::copy(device_current_storage, host_current_storage, host_current_storage.size(), 0, 0, stream);
}

void
AAHD_NonLocalDelayedBank::DownloadCurrentToHost(crb::Stream& stream)
{
  if (host_current_storage.size() > 0)
    crb::copy(host_current_storage, device_current_storage, host_current_storage.size(), 0, 0, stream);
}

void
AAHD_NonLocalDelayedBank::CopyCurrentDeviceToOldDevice(crb::Stream& stream)
{
  if (device_storage.size() > 0)
    crb::copy(device_storage, device_current_storage, device_storage.size(), 0, 0, stream);
}

void
AAHD_NonLocalDelayedBank::ZeroCurrentDevice()
{
  if (device_current_storage.size() > 0)
    device_current_storage.zero_fill();
}

AAHD_FLUDS::AAHD_FLUDS(unsigned int num_groups,
                       std::size_t num_angles,
                       const AAHD_FLUDSCommonData& common_data)
  : FLUDS(num_groups, num_angles, common_data.GetSPDS()),
    common_data_(common_data),
    local_psi_(common_data_.GetLocalNodeStackSize() * num_groups_and_angles_),
    nonlocal_incoming_psi_bank_(common_data_.GetNumNonLocalIncomingNodes(),
                                common_data_.GetNonLocalIncomingNodeOffsets(),
                                num_groups_and_angles_),
    nonlocal_outgoing_psi_bank_(common_data_.GetNumNonLocalOutgoingNodes(),
                                common_data_.GetNonLocalOutgoingNodeOffsets(),
                                num_groups_and_angles_)
{
  nonlocal_outgoing_psi_bank_.UpdateViews(deplocI_outgoing_psi_view_);
  nonlocal_incoming_psi_bank_.UpdateViews(prelocI_outgoing_psi_view_);
}

void
AAHD_FLUDS::AllocateDelayedLocalPsi()
{
  // FAS-edge delayed nodes: full stride (num_groups * num_angles)
  const std::size_t fas_count = common_data_.GetNumDelayedLocalNodes();
  delayed_local_psi_bank_ = AAHD_DelayedLocalBank(fas_count, num_groups_and_angles_);
  delayed_local_psi_view_ = std::span<double>(delayed_local_psi_bank_.host_storage);
  // Explicitly zero the device FAS_NEW buffer so that phantom slots (FAS entries for angles
  // that were promoted to AB for reversed FAS edges) remain zero and do not pollute GMRES.
  delayed_local_psi_bank_.UploadToDevice();
  delayed_local_psi_old_bank_ = AAHD_DelayedLocalBank(fas_count, num_groups_and_angles_);
  delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_bank_.host_storage);

  // Additionally-backward delayed nodes: compact stride (num_groups only, angle-independent)
  const std::size_t ab_count = common_data_.GetTotalDelayedLocalNodes() - fas_count;
  if (ab_count > 0)
  {
    ab_delayed_psi_bank_ = AAHD_Bank(ab_count * num_groups_);
    ab_delayed_psi_old_bank_ = AAHD_Bank(ab_count * num_groups_);
  }

  promoted_delayed_incoming_psi_bank_ =
    AAHD_NonLocalDelayedBank(common_data_.GetPromotedDelayedIncomingNodeSizes(),
                             common_data_.GetPromotedDelayedIncomingNodeOffsets(),
                             num_groups_);
  promoted_delayed_incoming_psi_bank_.UpdateViews(delayed_promoted_incoming_psi_view_,
                                                  delayed_promoted_incoming_psi_old_view_);

  promoted_delayed_outgoing_psi_bank_ =
    AAHD_NonLocalBank(common_data_.GetPromotedDelayedOutgoingNodeSizes(),
                      common_data_.GetPromotedDelayedOutgoingNodeOffsets(),
                      num_groups_);
  promoted_delayed_outgoing_psi_bank_.UpdateViews(promoted_delayed_outgoing_psi_view_);
}

void
AAHD_FLUDS::AllocateDelayedPrelocIOutgoingPsi()
{
  nonlocal_delayed_incoming_psi_bank_ =
    AAHD_NonLocalDelayedBank(common_data_.GetNumNonLocalDelayedIncomingNodes(),
                             common_data_.GetNonLocalDelayedIncomingNodeOffsets(),
                             num_groups_and_angles_);
  nonlocal_delayed_incoming_psi_bank_.UpdateViews(delayed_prelocI_outgoing_psi_view_,
                                                  delayed_prelocI_outgoing_psi_old_view_);
}

void
AAHD_FLUDS::AllocateSaveAngularFlux(DiscreteOrdinatesProblem& problem, const LBSGroupset& groupset)
{
  if (not problem.GetPsiNewLocal()[groupset.id].empty() && save_angular_flux_.IsNotInitialized())
  {
    auto* mesh_carrier_ptr = problem.GetMeshCarrier();
    save_angular_flux_ = AAHD_Bank(mesh_carrier_ptr->num_nodes_total * num_groups_and_angles_);
  }
}

void
AAHD_FLUDS::SetDelayedOutgoingPsiOldToNew()
{
  nonlocal_delayed_incoming_psi_bank_.SetOldToNew();
  promoted_delayed_incoming_psi_bank_.SetOldToNew();
}

void
AAHD_FLUDS::SetDelayedOutgoingPsiNewToOld()
{
  nonlocal_delayed_incoming_psi_bank_.SetNewToOld();
  promoted_delayed_incoming_psi_bank_.SetNewToOld();
}

void
AAHD_FLUDS::SetDelayedLocalPsiOldToNew()
{
  delayed_local_psi_bank_.host_storage = delayed_local_psi_old_bank_.host_storage;
  delayed_local_psi_view_ = std::span<double>(delayed_local_psi_bank_.host_storage);
  ab_delayed_psi_bank_.host_storage = ab_delayed_psi_old_bank_.host_storage;
}

void
AAHD_FLUDS::SetDelayedLocalPsiNewToOld()
{
  delayed_local_psi_old_bank_.host_storage = delayed_local_psi_bank_.host_storage;
  delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_bank_.host_storage);
  ab_delayed_psi_old_bank_.host_storage = ab_delayed_psi_bank_.host_storage;
}

void
AAHD_FLUDS::CopyDelayedPsiToDevice()
{
  if (common_data_.GetNumDelayedLocalNodes() > 0)
    delayed_local_psi_old_bank_.UploadToDevice(stream_);
  if (!ab_delayed_psi_old_bank_.IsNotInitialized())
    ab_delayed_psi_old_bank_.UploadToDevice(stream_);
  if (HasNonLocalDelayedIncomingPsi())
    nonlocal_delayed_incoming_psi_bank_.UploadToDevice(stream_);
  if (HasPromotedDelayedIncomingPsi())
    promoted_delayed_incoming_psi_bank_.UploadToDevice(stream_);
}

void
AAHD_FLUDS::CopyDelayedPsiNewToOldOnDevice()
{
  if (common_data_.GetNumDelayedLocalNodes() > 0)
    delayed_local_psi_old_bank_.CopyDeviceFrom(delayed_local_psi_bank_, stream_);
  if (!ab_delayed_psi_old_bank_.IsNotInitialized())
    ab_delayed_psi_old_bank_.CopyDeviceFrom(ab_delayed_psi_bank_, stream_);
  if (HasNonLocalDelayedIncomingPsi())
    nonlocal_delayed_incoming_psi_bank_.CopyCurrentDeviceToOldDevice(stream_);
  if (HasPromotedDelayedIncomingPsi())
    promoted_delayed_incoming_psi_bank_.CopyCurrentDeviceToOldDevice(stream_);
}

double
AAHD_FLUDS::ComputeDelayedPsiPointwiseChangeOnDevice()
{
  double local_max = 0.0;

  if (common_data_.GetNumDelayedLocalNodes() > 0)
    AccumulatePsiPointwiseChange(delayed_local_psi_old_bank_.device_storage,
                                 delayed_local_psi_bank_.device_storage,
                                 local_max,
                                 stream_);
  if (!ab_delayed_psi_old_bank_.IsNotInitialized())
    AccumulatePsiPointwiseChange(ab_delayed_psi_old_bank_.device_storage,
                                 ab_delayed_psi_bank_.device_storage,
                                 local_max,
                                 stream_);
  if (HasNonLocalDelayedIncomingPsi())
    AccumulatePsiPointwiseChange(nonlocal_delayed_incoming_psi_bank_.device_storage,
                                 nonlocal_delayed_incoming_psi_bank_.device_current_storage,
                                 local_max,
                                 stream_);
  if (HasPromotedDelayedIncomingPsi())
    AccumulatePsiPointwiseChange(promoted_delayed_incoming_psi_bank_.device_storage,
                                 promoted_delayed_incoming_psi_bank_.device_current_storage,
                                 local_max,
                                 stream_);

  return local_max;
}

void
AAHD_FLUDS::ZeroDelayedPsiNewOnDevice()
{
  if (common_data_.GetNumDelayedLocalNodes() > 0)
    delayed_local_psi_bank_.ZeroDevice();
  if (!ab_delayed_psi_bank_.IsNotInitialized())
    ab_delayed_psi_bank_.ZeroDevice();
  if (HasNonLocalDelayedIncomingPsi())
    nonlocal_delayed_incoming_psi_bank_.ZeroCurrentDevice();
  if (HasPromotedDelayedIncomingPsi())
    promoted_delayed_incoming_psi_bank_.ZeroCurrentDevice();
  if (HasPromotedDelayedOutgoingPsi())
    promoted_delayed_outgoing_psi_bank_.ZeroDevice();
}

void
AAHD_FLUDS::CopyNonLocalIncomingPsiToDevice()
{
  if (HasNonLocalIncomingPsi())
    nonlocal_incoming_psi_bank_.UploadToDevice(stream_);
}

AAHD_FLUDSPointerSet
AAHD_FLUDS::GetDevicePointerSet()
{
  AAHD_FLUDSPointerSet pointer_set;
  // local psi
  pointer_set.local_psi = local_psi_.get();
  if (common_data_.GetLocalNodeStackSize() > 0)
    assert(pointer_set.local_psi != nullptr);
  // FAS delayed local psi
  pointer_set.fas_count = common_data_.GetNumDelayedLocalNodes();
  pointer_set.delayed_local_psi = delayed_local_psi_bank_.device_storage.get();
  pointer_set.delayed_local_psi_old = delayed_local_psi_old_bank_.device_storage.get();
  if (pointer_set.fas_count > 0)
  {
    assert(pointer_set.delayed_local_psi != nullptr);
    assert(pointer_set.delayed_local_psi_old != nullptr);
  }
  // Additionally-backward compact delayed buffer
  pointer_set.groupset_size = num_groups_;
  pointer_set.ab_delayed_psi = ab_delayed_psi_bank_.device_storage.get();
  pointer_set.ab_delayed_psi_old = ab_delayed_psi_old_bank_.device_storage.get();
  // non-local psi
  pointer_set.nonlocal_incoming_psi = nonlocal_incoming_psi_bank_.device_storage.get();
  if (common_data_.GetNonLocalIncomingNodeOffsets().back() > 0)
  {
    assert(pointer_set.nonlocal_incoming_psi != nullptr);
  }
  pointer_set.nonlocal_delayed_incoming_psi_old =
    nonlocal_delayed_incoming_psi_bank_.device_storage.get();
  pointer_set.nonlocal_delayed_incoming_count =
    common_data_.GetNonLocalDelayedIncomingNodeOffsets().back();
  if (common_data_.GetNonLocalDelayedIncomingNodeOffsets().back() > 0)
  {
    assert(pointer_set.nonlocal_delayed_incoming_psi_old != nullptr);
  }
  pointer_set.promoted_delayed_incoming_psi =
    promoted_delayed_incoming_psi_bank_.CurrentDevicePointer();
  pointer_set.promoted_delayed_incoming_psi_old =
    promoted_delayed_incoming_psi_bank_.device_storage.get();
  if (common_data_.GetPromotedDelayedIncomingNodeOffsets().back() > 0)
  {
    assert(pointer_set.promoted_delayed_incoming_psi != nullptr);
    assert(pointer_set.promoted_delayed_incoming_psi_old != nullptr);
  }
  pointer_set.promoted_delayed_outgoing_psi =
    promoted_delayed_outgoing_psi_bank_.device_storage.get();
  if (common_data_.GetPromotedDelayedOutgoingNodeOffsets().back() > 0)
  {
    assert(pointer_set.promoted_delayed_outgoing_psi != nullptr);
  }
  pointer_set.nonlocal_outgoing_psi = nonlocal_outgoing_psi_bank_.device_storage.get();
  if (common_data_.GetNonLocalOutgoingNodeOffsets().back() > 0)
  {
    assert(pointer_set.nonlocal_outgoing_psi != nullptr);
  }
  // stride size
  pointer_set.stride_size = num_groups_and_angles_;
  return pointer_set;
}

double*
AAHD_FLUDS::DevicePrelocIOutgoingPsi(std::size_t loc, std::size_t block_pos)
{
  const auto offset = common_data_.GetNonLocalIncomingNodeOffsets()[loc] * num_groups_and_angles_;
  return nonlocal_incoming_psi_bank_.device_storage.get() + offset + block_pos;
}

double*
AAHD_FLUDS::DeviceDelayedPrelocIOutgoingPsi(std::size_t loc, std::size_t block_pos)
{
  const auto offset =
    common_data_.GetNonLocalDelayedIncomingNodeOffsets()[loc] * num_groups_and_angles_;
  return nonlocal_delayed_incoming_psi_bank_.CurrentDevicePointer() + offset + block_pos;
}

double*
AAHD_FLUDS::DeviceDeplocIOutgoingPsi(std::size_t loc, std::size_t block_pos)
{
  const auto offset = common_data_.GetNonLocalOutgoingNodeOffsets()[loc] * num_groups_and_angles_;
  return nonlocal_outgoing_psi_bank_.device_storage.get() + offset + block_pos;
}

double*
AAHD_FLUDS::DevicePromotedDelayedIncomingPsi(std::size_t loc, std::size_t block_pos)
{
  const auto offset = common_data_.GetPromotedDelayedIncomingNodeOffsets()[loc] * num_groups_;
  return promoted_delayed_incoming_psi_bank_.CurrentDevicePointer() + offset + block_pos;
}

double*
AAHD_FLUDS::DevicePromotedDelayedOutgoingPsi(std::size_t loc, std::size_t block_pos)
{
  const auto offset = common_data_.GetPromotedDelayedOutgoingNodeOffsets()[loc] * num_groups_;
  return promoted_delayed_outgoing_psi_bank_.device_storage.get() + offset + block_pos;
}

namespace
{

void
ValidateDeviceBuffer(const char* name,
                     const double* ptr,
                     std::size_t storage_size,
                     std::size_t offset,
                     std::size_t size)
{
  OpenSnLogicalErrorIf(size > 0 and ptr == nullptr, std::string(name) + ": null device buffer.");
  OpenSnLogicalErrorIf(offset + size > storage_size,
                       std::string(name) + ": message range exceeds device buffer. offset=" +
                         std::to_string(offset) + ", size=" + std::to_string(size) +
                         ", storage_size=" + std::to_string(storage_size));
}

} // namespace

void
AAHD_FLUDS::ValidateDevicePrelocIOutgoingPsi(std::size_t loc,
                                             std::size_t block_pos,
                                             std::size_t size) const
{
  const auto offset = common_data_.GetNonLocalIncomingNodeOffsets()[loc] * num_groups_and_angles_ +
                      block_pos;
  ValidateDeviceBuffer("AAHD_FLUDS::DevicePrelocIOutgoingPsi",
                       nonlocal_incoming_psi_bank_.device_storage.get(),
                       nonlocal_incoming_psi_bank_.device_storage.size(),
                       offset,
                       size);
}

void
AAHD_FLUDS::ValidateDeviceDelayedPrelocIOutgoingPsi(std::size_t loc,
                                                    std::size_t block_pos,
                                                    std::size_t size) const
{
  const auto offset =
    common_data_.GetNonLocalDelayedIncomingNodeOffsets()[loc] * num_groups_and_angles_ + block_pos;
  ValidateDeviceBuffer("AAHD_FLUDS::DeviceDelayedPrelocIOutgoingPsi",
                       nonlocal_delayed_incoming_psi_bank_.device_current_storage.get(),
                       nonlocal_delayed_incoming_psi_bank_.device_current_storage.size(),
                       offset,
                       size);
}

void
AAHD_FLUDS::ValidateDeviceDeplocIOutgoingPsi(std::size_t loc,
                                             std::size_t block_pos,
                                             std::size_t size) const
{
  const auto offset = common_data_.GetNonLocalOutgoingNodeOffsets()[loc] * num_groups_and_angles_ +
                      block_pos;
  ValidateDeviceBuffer("AAHD_FLUDS::DeviceDeplocIOutgoingPsi",
                       nonlocal_outgoing_psi_bank_.device_storage.get(),
                       nonlocal_outgoing_psi_bank_.device_storage.size(),
                       offset,
                       size);
}

void
AAHD_FLUDS::ValidateDevicePromotedDelayedIncomingPsi(std::size_t loc,
                                                     std::size_t block_pos,
                                                     std::size_t size) const
{
  const auto offset = common_data_.GetPromotedDelayedIncomingNodeOffsets()[loc] * num_groups_ +
                      block_pos;
  ValidateDeviceBuffer("AAHD_FLUDS::DevicePromotedDelayedIncomingPsi",
                       promoted_delayed_incoming_psi_bank_.device_current_storage.get(),
                       promoted_delayed_incoming_psi_bank_.device_current_storage.size(),
                       offset,
                       size);
}

void
AAHD_FLUDS::ValidateDevicePromotedDelayedOutgoingPsi(std::size_t loc,
                                                     std::size_t block_pos,
                                                     std::size_t size) const
{
  const auto offset = common_data_.GetPromotedDelayedOutgoingNodeOffsets()[loc] * num_groups_ +
                      block_pos;
  ValidateDeviceBuffer("AAHD_FLUDS::DevicePromotedDelayedOutgoingPsi",
                       promoted_delayed_outgoing_psi_bank_.device_storage.get(),
                       promoted_delayed_outgoing_psi_bank_.device_storage.size(),
                       offset,
                       size);
}

void
AAHD_FLUDS::CopyNonLocalOutgoingPsiFromDevice()
{
  if (HasNonLocalOutgoingPsi())
    nonlocal_outgoing_psi_bank_.DownloadToHost(stream_);
}

void
AAHD_FLUDS::CopyLocalDelayedPsiFromDevice()
{
  if (common_data_.GetNumDelayedLocalNodes() > 0)
    delayed_local_psi_bank_.DownloadToHost(stream_);
  if (!ab_delayed_psi_bank_.IsNotInitialized())
    ab_delayed_psi_bank_.DownloadToHost(stream_);
}

void
AAHD_FLUDS::CopyPromotedDelayedOutgoingPsiFromDevice()
{
  if (HasPromotedDelayedOutgoingPsi())
    promoted_delayed_outgoing_psi_bank_.DownloadToHost(stream_);
}

void
AAHD_FLUDS::CopyPsiFromDevice()
{
  if (HasNonLocalOutgoingPsi())
    nonlocal_outgoing_psi_bank_.DownloadToHost(stream_);
  if (common_data_.GetNumDelayedLocalNodes() > 0)
    delayed_local_psi_bank_.DownloadToHost(stream_);
  if (!ab_delayed_psi_bank_.IsNotInitialized())
    ab_delayed_psi_bank_.DownloadToHost(stream_);
}

void
AAHD_FLUDS::CopySaveAngularFluxFromDevice()
{
  if (HasSaveAngularFlux())
    save_angular_flux_.DownloadToHost(stream_);
}

void
AAHD_FLUDS::CopySaveAngularFluxToDestinationPsi(DiscreteOrdinatesProblem& problem,
                                                const LBSGroupset& groupset,
                                                AngleSet& angle_set)
{
  if (not HasSaveAngularFlux())
    return;

  // loop for each cell in the mesh
  auto* mesh_carrier = problem.GetMeshCarrier();
  auto grid = problem.GetGrid();
  auto& destination_psi = problem.GetPsiNewLocal()[groupset.id];
  const auto& discretization = problem.GetSpatialDiscretization();
  std::size_t groupset_angle_group_stride =
    groupset.psi_uk_man_.GetNumberOfUnknowns() * groupset.GetNumGroups();
  for (const Cell& cell : grid->local_cells)
  {
    // get pointer to the cell's angular fluxes
    double* dst_psi =
      &destination_psi[discretization.MapDOFLocal(cell, 0, groupset.psi_uk_man_, 0, 0)];
    double* src_psi = save_angular_flux_.host_storage.data() +
                      mesh_carrier->saved_psi_offset[cell.local_id] * num_groups_and_angles_;
    // get number of cell nodes
    std::uint32_t cell_num_nodes = discretization.GetCellMapping(cell).GetNumNodes();
    // loop for each cell node
    for (std::uint32_t i = 0; i < cell_num_nodes; ++i)
    {
      // loop for each angle
      for (std::uint32_t as_ss_idx = 0; as_ss_idx < angle_set.GetNumAngles(); ++as_ss_idx)
      {
        auto direction_num = angle_set.GetAngleIndices()[as_ss_idx];
        // compute dst and src corresponding to the direction
        double* dst = dst_psi + direction_num * num_groups_;
        double* src = src_psi + as_ss_idx * num_groups_;
        // copy the flux for each group
        std::memcpy(dst, src, num_groups_ * sizeof(double));
      }
      // move src and dts to next node
      dst_psi += groupset_angle_group_stride;
      src_psi += num_groups_and_angles_;
    }
  }
}

} // namespace opensn
