// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include <cstring>
#include <numeric>

namespace opensn
{
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
  const std::size_t fas_count = common_data_.GetNumDelayedLocalNodes();
  delayed_local_psi_bank_ = AAHD_DelayedLocalBank(fas_count, num_groups_and_angles_);
  delayed_local_psi_view_ = std::span<double>(delayed_local_psi_bank_.host_storage);
  delayed_local_psi_bank_.UploadToDevice();
  delayed_local_psi_old_bank_ = AAHD_DelayedLocalBank(fas_count, num_groups_and_angles_);
  delayed_local_psi_old_view_ = std::span<double>(delayed_local_psi_old_bank_.host_storage);

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
  if (not ab_delayed_psi_old_bank_.IsNotInitialized())
    ab_delayed_psi_old_bank_.UploadToDevice(stream_);
  if (HasNonLocalDelayedIncomingPsi())
    nonlocal_delayed_incoming_psi_bank_.UploadToDevice(stream_);
  if (HasPromotedDelayedIncomingPsi())
    promoted_delayed_incoming_psi_bank_.UploadToDevice(stream_);
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
  pointer_set.fas_count = common_data_.GetNumDelayedLocalNodes();
  pointer_set.delayed_local_psi = delayed_local_psi_bank_.device_storage.get();
  pointer_set.delayed_local_psi_old = delayed_local_psi_old_bank_.device_storage.get();
  if (pointer_set.fas_count > 0)
  {
    assert(pointer_set.delayed_local_psi != nullptr);
    assert(pointer_set.delayed_local_psi_old != nullptr);
  }
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
  pointer_set.promoted_delayed_outgoing_psi =
    promoted_delayed_outgoing_psi_bank_.device_storage.get();
  pointer_set.nonlocal_outgoing_psi = nonlocal_outgoing_psi_bank_.device_storage.get();
  if (common_data_.GetNonLocalOutgoingNodeOffsets().back() > 0)
  {
    assert(pointer_set.nonlocal_outgoing_psi != nullptr);
  }
  // stride size
  pointer_set.stride_size = num_groups_and_angles_;
  return pointer_set;
}

void
AAHD_FLUDS::CopyPsiFromDevice()
{
  if (HasNonLocalOutgoingPsi())
    nonlocal_outgoing_psi_bank_.DownloadToHost(stream_);
  if (common_data_.GetNumDelayedLocalNodes() > 0)
    delayed_local_psi_bank_.DownloadToHost(stream_);
  if (not ab_delayed_psi_bank_.IsNotInitialized())
    ab_delayed_psi_bank_.DownloadToHost(stream_);
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
  if (not ab_delayed_psi_bank_.IsNotInitialized())
    ab_delayed_psi_bank_.DownloadToHost(stream_);
}

void
AAHD_FLUDS::CopyPromotedDelayedOutgoingPsiFromDevice()
{
  if (HasPromotedDelayedOutgoingPsi())
    promoted_delayed_outgoing_psi_bank_.DownloadToHost(stream_);
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
