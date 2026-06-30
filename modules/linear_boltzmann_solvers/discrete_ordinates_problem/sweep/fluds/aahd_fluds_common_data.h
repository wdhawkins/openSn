// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_structs.h"
#include "framework/data_types/vector3.h"
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace opensn
{

class AAHD_AngleSet;
class SpatialDiscretization;
class SweepBoundary;

/**
 * Random-access stack based FLUDS.
 */
class AAHD_FLUDSCommonData : public FLUDSCommonData
{
public:
  /**
   * Constructor from SPDS and grid nodal mappings.
   * Initialize the index tracker for all face nodes of the grid. Once created, the indexes
   * are utilized by the AAHD_FLUDS to access the correct location in the flux storage arrays.
   */
  AAHD_FLUDSCommonData(const SPDS& spds,
                       const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
                       const SpatialDiscretization& sdm,
                       std::vector<Vector3> all_omegas = {});

  /// Get constant reference to the face node tracker map.
  const std::map<FaceNode, AAHD_NodeIndex>& GetNodeTracker() const { return node_tracker_; }

  /// \name Size getters
  /// \{
  /// Get size of the local face node stack.
  std::size_t GetLocalNodeStackSize() const { return local_node_stack_size_; }
  /// Get size of the delayed local face node stack.
  std::size_t GetNumDelayedLocalNodes() const { return delayed_local_node_stack_size_; }
  /// Get sizes of the incoming non-local face node banks.
  const std::vector<std::size_t>& GetNumNonLocalIncomingNodes() const
  {
    return nonlocal_incoming_node_sizes_;
  }
  /// Get offsets of the incoming non-local face node banks for each location.
  const std::vector<std::size_t>& GetNonLocalIncomingNodeOffsets() const
  {
    return nonlocal_incoming_node_offsets_;
  }
  /// Get sizes of the delayed incoming non-local face node banks.
  const std::vector<std::size_t>& GetNumNonLocalDelayedIncomingNodes() const
  {
    return nonlocal_delayed_incoming_node_sizes_;
  }
  /// Get offsets of the delayed incoming non-local face node banks for each location.
  const std::vector<std::size_t>& GetNonLocalDelayedIncomingNodeOffsets() const
  {
    return nonlocal_delayed_incoming_node_offsets_;
  }
  /// Get sizes of the outgoing non-local face node banks.
  const std::vector<std::size_t>& GetNumNonLocalOutgoingNodes() const
  {
    return nonlocal_outgoing_node_sizes_;
  }
  /// Get offsets of the outgoing non-local face node banks for each location.
  const std::vector<std::size_t>& GetNonLocalOutgoingNodeOffsets() const
  {
    return nonlocal_outgoing_node_offsets_;
  }
  /// \}

  /// Get pointer to indexes on device.
  const std::uint64_t* GetDeviceIndex() const { return device_node_indexes_; }
  const std::uint64_t* const* GetDeviceIndexPtrArray() const { return device_index_ptr_array_; }
  std::size_t GetTotalDelayedLocalNodes() const { return total_delayed_local_node_stack_size_; }
  const std::vector<int>& GetPromotedDelayedIncomingLocations() const
  {
    return promoted_delayed_incoming_locations_;
  }
  const std::vector<std::size_t>& GetPromotedDelayedIncomingNodeSizes() const
  {
    return promoted_delayed_incoming_node_sizes_;
  }
  const std::vector<std::size_t>& GetPromotedDelayedIncomingNodeOffsets() const
  {
    return promoted_delayed_incoming_node_offsets_;
  }
  const std::vector<int>& GetPromotedDelayedOutgoingLocations() const
  {
    return promoted_delayed_outgoing_locations_;
  }
  const std::vector<std::size_t>& GetPromotedDelayedOutgoingNodeSizes() const
  {
    return promoted_delayed_outgoing_node_sizes_;
  }
  const std::vector<std::size_t>& GetPromotedDelayedOutgoingNodeOffsets() const
  {
    return promoted_delayed_outgoing_node_offsets_;
  }

  /// Append an associated angle set pointer.
  void AddAssociatedAngleSet(AAHD_AngleSet* as) const { associated_anglesets_.push_back(as); }

  /// Update node tarcker with boundary data and transfer all data to device.
  void UpdateBoundaryAndSyncWithDevice(
    const SpatialDiscretization& sdm,
    const std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  /// Destructor.
  ~AAHD_FLUDSCommonData() override;

protected:
  /// Map face node to its associated index in the corresponding bank.
  std::map<FaceNode, AAHD_NodeIndex> node_tracker_;
  /// List of anglesets associated to this sweep ordering.
  mutable std::vector<AAHD_AngleSet*> associated_anglesets_;

  /// \name Allocation sizes
  /// \{
  /// Size of the local face node stack.
  std::size_t local_node_stack_size_ = 0;
  /// Size of the delayed local face node vector.
  std::size_t delayed_local_node_stack_size_ = 0;
  /// Size of the incoming non-local face nodes.
  std::vector<std::size_t> nonlocal_incoming_node_sizes_;
  /// Offset of each location to its incoming non-local face nodes.
  std::vector<std::size_t> nonlocal_incoming_node_offsets_;
  /// Size of the delayed incoming non-local face nodes.
  std::vector<std::size_t> nonlocal_delayed_incoming_node_sizes_;
  /// Offset of each location to its delayed incoming non-local face nodes.
  std::vector<std::size_t> nonlocal_delayed_incoming_node_offsets_;
  /// Size of the outgoing non-local face nodes.
  std::vector<std::size_t> nonlocal_outgoing_node_sizes_;
  /// Offset of each location to its outgoing non-local face nodes.
  std::vector<std::size_t> nonlocal_outgoing_node_offsets_;
  std::vector<int> promoted_delayed_incoming_locations_;
  std::vector<std::size_t> promoted_delayed_incoming_node_sizes_;
  std::vector<std::size_t> promoted_delayed_incoming_node_offsets_;
  std::vector<int> promoted_delayed_outgoing_locations_;
  std::vector<std::size_t> promoted_delayed_outgoing_node_sizes_;
  std::vector<std::size_t> promoted_delayed_outgoing_node_offsets_;
  std::vector<Vector3> all_omegas_;
  std::map<std::pair<std::size_t, FaceNode>, std::uint64_t> promoted_delayed_incoming_indices_;
  std::map<std::pair<std::size_t, FaceNode>, std::uint64_t> promoted_delayed_outgoing_indices_;
  std::unordered_map<FaceNode, std::pair<bool, bool>, FaceNodeHash> boundary_node_flags_;
  /// \}

  /// \name Device storage for node indexes
  /// \{
  /// Device storage for node indexes.
  std::uint64_t* device_node_indexes_ = nullptr;
  std::vector<std::uint64_t*> device_per_angle_indexes_;
  std::uint64_t** device_index_ptr_array_ = nullptr;
  std::size_t total_delayed_local_node_stack_size_ = 0;
  /// Construct flatten node index structure and copy it to device.
  void CopyFlattenNodeIndexToDevice(const SpatialDiscretization& sdm);
  /// \}

  /// \name Index construction methods for each type of face nodes
  /// \{
  /// Compute the local face node stack for non delayed local face nodes.
  void ComputeNodeIndexForNonDelayedLocalFaces(const SpatialDiscretization& sdm);
  /// Compute the indexes for FAS face nodes.
  void ComputeNodeIndexForDelayedLocalFaces(const SpatialDiscretization& sdm);
  /// Compute the indexes for non-local face nodes.
  void ComputeNodeIndexForNonLocalFaces(const SpatialDiscretization& sdm);
  void ComputePromotedDelayedNonLocalFaces(const SpatialDiscretization& sdm);
  /// Compute the index for parallel faces.
  void ComputeNodeIndexForParallelFaces(const SpatialDiscretization& sdm);
  /// Compute the indexes for boundary face nodes.
  void ComputeNodeIndexForBoundaryFaces(
    const SpatialDiscretization& sdm,
    const std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);
  /// \}

  /// Deallocate memory for flatten node indexes on device.
  void DeallocateDeviceMemory();
};

} // namespace opensn
