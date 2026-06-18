// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caribou/main.hpp"
#include <cstdint>
#include <span>
#include <utility>

namespace crb = caribou;

namespace opensn
{

class DiscreteOrdinatesProblem;

/// Host-device bank storage structure for device-transportable FLUDS.
struct AAHD_Bank
{
  AAHD_Bank() = default;
  /// Construct and allocate memory.
  AAHD_Bank(std::size_t size) : host_storage(size, 0.0), device_storage(size) {}

  AAHD_Bank(const AAHD_Bank& other);
  AAHD_Bank& operator=(const AAHD_Bank& other);
  AAHD_Bank(AAHD_Bank&& other) = default;
  AAHD_Bank& operator=(AAHD_Bank&& other) = default;

  /// Clear host and device storage.
  void Clear();
  /// Upload data from host to device.
  void UploadToDevice();
  /// Upload data from host to device asynchronously.
  void UploadToDevice(crb::Stream& stream);
  /// Download data from device to host.
  void DownloadToHost();
  /// Download data from device to host asynchronously.
  void DownloadToHost(crb::Stream& stream);
  /// Copy data from another device bank asynchronously.
  void CopyDeviceFrom(const AAHD_Bank& other, crb::Stream& stream);
  /// Zero device storage.
  void ZeroDevice();

  /// Check if the bank is not initialized.
  bool IsNotInitialized() const { return device_storage.get() == nullptr; }

  /// Host storage for the bank.
  crb::HostVector<double> host_storage;
  /// Device storage for the bank.
  crb::DeviceMemory<double> device_storage;
};

/// Delayed local bank storage structure.
struct AAHD_DelayedLocalBank : public AAHD_Bank
{
  AAHD_DelayedLocalBank() = default;
  /// Member constructor.
  AAHD_DelayedLocalBank(std::size_t size, std::size_t stride_size) : AAHD_Bank(size * stride_size)
  {
  }
};

/// Non-local bank storage structure.
struct AAHD_NonLocalBank : public AAHD_Bank
{
  AAHD_NonLocalBank() = default;
  /// Member constructor.
  AAHD_NonLocalBank(const std::vector<std::size_t>& loc_sizes,
                    const std::vector<std::size_t>& loc_offsets,
                    std::size_t stride);

  /// Update views.
  void UpdateViews(std::vector<std::span<double>>& views);

  /// Reference to the sizes of each location.
  const std::vector<std::size_t>* location_sizes = nullptr;
  /// Reference to the offsets of each location.
  const std::vector<std::size_t>* location_offsets = nullptr;
  /// Stride size.
  std::size_t stride_size;
};

/// Non-local delayed bank storage structure.
struct AAHD_NonLocalDelayedBank : public AAHD_NonLocalBank
{
  AAHD_NonLocalDelayedBank() = default;
  /// Member constructor.
  AAHD_NonLocalDelayedBank(const std::vector<std::size_t>& loc_sizes,
                           const std::vector<std::size_t>& loc_offsets,
                           std::size_t stride);

  /// Update views.
  void UpdateViews(std::vector<std::span<double>>& current_delayed_views,
                   std::vector<std::span<double>>& old_delayed_views);
  /// Set current delayed views to old delayed views.
  void SetNewToOld();
  /// Set old delayed views to current delayed views.
  void SetOldToNew();
  /// Upload current delayed data from host to device.
  void UploadCurrentToDevice(crb::Stream& stream);
  /// Download current delayed data from device to host.
  void DownloadCurrentToHost(crb::Stream& stream);
  /// Copy current delayed device data to old delayed device data.
  void CopyCurrentDeviceToOldDevice(crb::Stream& stream);
  /// Zero current delayed device data.
  void ZeroCurrentDevice();
  /// Pointer to current delayed device data.
  double* CurrentDevicePointer() { return device_current_storage.get(); }

  /// Host storage for current delayed bank.
  crb::HostVector<double> host_current_storage;
  /// Device storage for current delayed bank.
  crb::DeviceMemory<double> device_current_storage;
};

/// AAH FLUDS for device.
class AAHD_FLUDS : public FLUDS
{
public:
  /// \name Constructor
  /// \{
  /// Construct and allocate memory for the FLUDS on both the host and device.
  AAHD_FLUDS(unsigned int num_groups,
             std::size_t num_angles,
             const AAHD_FLUDSCommonData& common_data);
  /// \}

  /// \name Deallocate banks
  /// \{
  /// Clear local and non-local incoming banks.
  void ClearLocalAndReceivePsi() override {}
  /// Clear non-local outgoing banks.
  void ClearSendPsi() override {}
  /// \}

  /// \name Allocate memory for banks
  /// \{
  /// Allocate internal local psi storage.
  void AllocateInternalLocalPsi() override {}
  /// Allocate delayed local psi storage (both old and new).
  void AllocateDelayedLocalPsi() override;
  /// Allocate non-local incoming psi storage.
  void AllocatePrelocIOutgoingPsi() override {}
  /// Allocate non-local delayed incoming psi storage.
  void AllocateDelayedPrelocIOutgoingPsi() override;
  /// Allocate non-local outgoing psi storage.
  void AllocateOutgoingPsi() override {}
  /// Allocate memory for save angular flux if needed.
  void AllocateSaveAngularFlux(DiscreteOrdinatesProblem& problem, const LBSGroupset& groupset);
  /// \}

  /// \name Size getters
  /// \{
  /// Get size of non-local incoming bank.
  inline std::size_t GetNonLocalIncomingNumUnknowns(std::size_t loc) const
  {
    return common_data_.GetNumNonLocalIncomingNodes()[loc] * num_groups_and_angles_;
  }
  /// Get size of non-local delayed incoming bank
  inline std::size_t GetNonLocalDelayedIncomingNumUnknowns(std::size_t loc) const
  {
    return common_data_.GetNumNonLocalDelayedIncomingNodes()[loc] * num_groups_and_angles_;
  }
  /// Get size of non-local outgoing bank
  inline std::size_t GetNonLocalOutgoingNumUnknowns(std::size_t loc) const
  {
    return common_data_.GetNumNonLocalOutgoingNodes()[loc] * num_groups_and_angles_;
  }
  /// Get number of groups by number of angles
  inline std::size_t GetStrideSize() const { return num_groups_and_angles_; }
  /// Get size of compact promoted cross-rank delayed incoming bank.
  inline std::size_t GetPromotedDelayedIncomingNumUnknowns(std::size_t loc) const
  {
    return common_data_.GetPromotedDelayedIncomingNodeSizes()[loc] * num_groups_;
  }
  /// Get size of compact promoted cross-rank delayed outgoing bank.
  inline std::size_t GetPromotedDelayedOutgoingNumUnknowns(std::size_t loc) const
  {
    return common_data_.GetPromotedDelayedOutgoingNodeSizes()[loc] * num_groups_;
  }
  /// \}

  /// \name Copy delayed fluxes
  /// \{
  /// Set non-local delayed incoming old psi to new psi.
  void SetDelayedOutgoingPsiOldToNew() override;
  /// Set non-local delayed incoming new psi to old psi.
  void SetDelayedOutgoingPsiNewToOld() override;
  /// Set delayed local old psi to new psi.
  void SetDelayedLocalPsiOldToNew() override;
  /// Set delayed local new psi to old psi.
  void SetDelayedLocalPsiNewToOld() override;
  /// \}

  /// \name Sweep preparation and clean up
  /// \{
  /// Copy delayed local and delayed non-local incoming psi to device.
  void CopyDelayedPsiToDevice();
  /// Copy delayed angular output/current device banks to old device banks.
  void CopyDelayedPsiNewToOldOnDevice();
  /// Compute device-side pointwise relative change for delayed angular fluxes.
  double ComputeDelayedPsiPointwiseChangeOnDevice();
  /// Zero delayed angular output/current device banks before a repeated sweep pass.
  void ZeroDelayedPsiNewOnDevice();
  /// Copy non-local incoming psi to device.
  void CopyNonLocalIncomingPsiToDevice();
  /// Get device pointers for each bank in FLUDS.
  AAHD_FLUDSPointerSet GetDevicePointerSet();
  /// Copy non-local outgoing and delayed local psi from device to host.
  void CopyPsiFromDevice();
  /// Copy save angular flux from device to host.
  void CopySaveAngularFluxFromDevice();
  /// Copy save angular flux from host contiguous buffer to destination psi.
  void CopySaveAngularFluxToDestinationPsi(DiscreteOrdinatesProblem& problem,
                                           const LBSGroupset& groupset,
                                           AngleSet& angle_set);
  /// \}

  /// Returns a span over the new (outgoing) AB delayed psi on host. Empty if no AB nodes.
  std::span<double> ABDelayedPsi() override
  {
    return std::span<double>(ab_delayed_psi_bank_.host_storage);
  }
  /// Returns a span over the old (incoming) AB delayed psi on host. Empty if no AB nodes.
  std::span<double> ABDelayedPsiOld() override
  {
    return std::span<double>(ab_delayed_psi_old_bank_.host_storage);
  }
  /// Returns compact promoted cross-rank delayed incoming psi on host.
  std::span<double> PromotedDelayedPsi() override
  {
    return std::span<double>(promoted_delayed_incoming_psi_bank_.host_current_storage);
  }
  /// Returns compact old promoted cross-rank delayed incoming psi on host.
  std::span<double> PromotedDelayedPsiOld() override
  {
    return std::span<double>(promoted_delayed_incoming_psi_bank_.host_storage);
  }
  /// Returns true if this FLUDS has additionally-backward delayed nodes.
  bool HasABNodes() const { return !ab_delayed_psi_old_bank_.IsNotInitialized(); }
  bool HasNonLocalIncomingPsi() const
  {
    return common_data_.GetNonLocalIncomingNodeOffsets().back() > 0;
  }
  bool HasNonLocalDelayedIncomingPsi() const
  {
    return common_data_.GetNonLocalDelayedIncomingNodeOffsets().back() > 0;
  }
  bool HasPromotedDelayedIncomingPsi() const
  {
    return common_data_.GetPromotedDelayedIncomingNodeOffsets().back() > 0;
  }
  bool HasNonLocalOutgoingPsi() const
  {
    return common_data_.GetNonLocalOutgoingNodeOffsets().back() > 0;
  }
  bool HasPromotedDelayedOutgoingPsi() const
  {
    return common_data_.GetPromotedDelayedOutgoingNodeOffsets().back() > 0;
  }
  bool HasAnyLocalDelayedPsi() const
  {
    return common_data_.GetNumDelayedLocalNodes() > 0 or HasABNodes();
  }
  /// Download only the non-local outgoing psi bank to host (for early MPI send).
  void CopyNonLocalOutgoingPsiFromDevice();
  /// Download only the local delayed psi banks (FAS + AB) to host.
  void CopyLocalDelayedPsiFromDevice();
  /// Download compact promoted cross-rank delayed outgoing psi from device.
  void CopyPromotedDelayedOutgoingPsiFromDevice();

  double* DevicePrelocIOutgoingPsi(std::size_t loc, std::size_t block_pos = 0);
  double* DeviceDelayedPrelocIOutgoingPsi(std::size_t loc, std::size_t block_pos = 0);
  double* DeviceDeplocIOutgoingPsi(std::size_t loc, std::size_t block_pos = 0);
  double* DevicePromotedDelayedIncomingPsi(std::size_t loc, std::size_t block_pos = 0);
  double* DevicePromotedDelayedOutgoingPsi(std::size_t loc, std::size_t block_pos = 0);
  void ValidateDevicePrelocIOutgoingPsi(std::size_t loc, std::size_t block_pos, std::size_t size) const;
  void ValidateDeviceDelayedPrelocIOutgoingPsi(std::size_t loc,
                                               std::size_t block_pos,
                                               std::size_t size) const;
  void ValidateDeviceDeplocIOutgoingPsi(std::size_t loc, std::size_t block_pos, std::size_t size) const;
  void ValidateDevicePromotedDelayedIncomingPsi(std::size_t loc,
                                                std::size_t block_pos,
                                                std::size_t size) const;
  void ValidateDevicePromotedDelayedOutgoingPsi(std::size_t loc,
                                                std::size_t block_pos,
                                                std::size_t size) const;

  std::vector<std::span<double>>& PromotedDelayedIncomingPsi()
  {
    return delayed_promoted_incoming_psi_view_;
  }
  std::vector<std::span<double>>& PromotedDelayedIncomingPsiOld()
  {
    return delayed_promoted_incoming_psi_old_view_;
  }
  std::vector<std::span<double>>& PromotedDelayedOutgoingPsi()
  {
    return promoted_delayed_outgoing_psi_view_;
  }

  /// Get reference to the common data
  const AAHD_FLUDSCommonData& GetCommonData() { return common_data_; }
  /// Get reference to stream.
  crb::Stream& GetStream() { return stream_; }
  /// Get saved angular flux device pointer.
  double* GetSavedAngularFluxDevicePointer() { return save_angular_flux_.device_storage.get(); }
  /// Check if the FLUDS has save angular flux storage.
  bool HasSaveAngularFlux() const { return not save_angular_flux_.host_storage.empty(); }

protected:
  /// Reference to the common data.
  const AAHD_FLUDSCommonData& common_data_;

  /// Device storage of local angular fluxes.
  crb::DeviceMemory<double> local_psi_;

  /// Delayed local bank for angular fluxes (FAS edges only; stride = num_groups_and_angles_).
  AAHD_DelayedLocalBank delayed_local_psi_bank_;
  /// Delayed local bank for old angular fluxes (FAS edges only; stride = num_groups_and_angles_).
  AAHD_DelayedLocalBank delayed_local_psi_old_bank_;

  /// Compact additionally-backward delayed bank (stride = num_groups_, not num_groups_and_angles_).
  AAHD_Bank ab_delayed_psi_bank_;
  AAHD_Bank ab_delayed_psi_old_bank_;

  /// Non-local bank for incoming angular fluxes.
  AAHD_NonLocalBank nonlocal_incoming_psi_bank_;
  /// Non-local bank for delayed incoming angular fluxes.
  AAHD_NonLocalDelayedBank nonlocal_delayed_incoming_psi_bank_;
  /// Non-local bank for outgoing angular fluxes.
  AAHD_NonLocalBank nonlocal_outgoing_psi_bank_;

  /// Compact promoted cross-rank delayed incoming angular fluxes.
  AAHD_NonLocalDelayedBank promoted_delayed_incoming_psi_bank_;
  std::vector<std::span<double>> delayed_promoted_incoming_psi_view_;
  std::vector<std::span<double>> delayed_promoted_incoming_psi_old_view_;
  /// Compact promoted cross-rank delayed outgoing angular fluxes.
  AAHD_NonLocalBank promoted_delayed_outgoing_psi_bank_;
  std::vector<std::span<double>> promoted_delayed_outgoing_psi_view_;

  /// Storage for saved angular fluxes.
  AAHD_Bank save_angular_flux_;

  /// Stream for asynchronous operations.
  crb::Stream stream_ = crb::Stream::get_null_stream();
};

} // namespace opensn
