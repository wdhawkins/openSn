// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds_common_data.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "framework/math/spatial_discretization/spatial_discretization.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caribou/main.hpp"
#include <cinttypes>
#include <map>
#include <set>
#include <tuple>

namespace crb = caribou;

namespace opensn
{

void
AAHD_FLUDSCommonData::UpdateBoundaryAndSyncWithDevice(
  const SpatialDiscretization& sdm,
  const std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries)
{
  ComputeNodeIndexForBoundaryFaces(sdm, boundaries);
  CopyFlattenNodeIndexToDevice(sdm);
  for (auto angleset : associated_anglesets_)
  {
    angleset->CopyBoundaryOffsetToDevice();
  }
}

void
AAHD_FLUDSCommonData::ComputeNodeIndexForBoundaryFaces(
  const SpatialDiscretization& sdm,
  const std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries)
{
  std::uint64_t incremental_boundary_index = 0;
  boundary_node_flags_.clear();
  const MeshContinuum& grid = *(spds_.GetGrid());
  for (const Cell& cell : grid.local_cells)
  {
    for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
    {
      const CellFace& face = cell.faces[f];
      const FaceOrientation& orientation = spds_.GetCellFaceOrientations()[cell.local_id][f];
      std::uint32_t num_face_nodes = sdm.GetCellMapping(cell).GetNumFaceNodes(f);

      if (face.has_neighbor)
        continue;

      auto& boundary = *(boundaries.at(face.neighbor_id));
      bool any_angle_incoming = false;
      for (const auto& omega : all_omegas_)
        any_angle_incoming = any_angle_incoming or (omega.Dot(face.normal) < 0.0);
      for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
      {
        FaceNode fn(cell.local_id, f, fnode);
        boundary_node_flags_[fn] = {boundary.IsReflecting(), boundary.IsAngleDependent()};
      }
      if (orientation == FaceOrientation::INCOMING)
      {
        for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
        {
          FaceNode fn(cell.local_id, f, fnode);
          for (auto angleset : associated_anglesets_)
          {
            std::uint64_t offset = boundary.GetOffsetToAngleset(fn, *angleset, false);
            angleset->AddBoundaryOffset(offset);
          }
          node_tracker_[fn] = AAHD_NodeIndex(incremental_boundary_index,
                                             false,
                                             true,
                                             boundary.IsReflecting(),
                                             boundary.IsAngleDependent());
          ++incremental_boundary_index;
        }
      }
      else if (any_angle_incoming and not boundary.IsReflecting())
      {
        for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
        {
          FaceNode fn(cell.local_id, f, fnode);
          for (auto angleset : associated_anglesets_)
          {
            const std::uint64_t offset = boundary.GetOffsetToAngleset(fn, *angleset, false);
            angleset->AddBoundaryOffset(offset);
          }
          node_tracker_[fn] =
            AAHD_NodeIndex(incremental_boundary_index, true, true, false, false);
          ++incremental_boundary_index;
        }
      }
      else if (orientation == FaceOrientation::OUTGOING && boundary.IsReflecting())
      {
        for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
        {
          FaceNode fn(cell.local_id, f, fnode);
          for (auto angleset : associated_anglesets_)
          {
            std::uint64_t offset = boundary.GetOffsetToAngleset(fn, *angleset, true);
            angleset->AddBoundaryOffset(offset);
          }
          node_tracker_[fn] = AAHD_NodeIndex(incremental_boundary_index, true, true, true, true);
          ++incremental_boundary_index;
        }
      }
      else
      {
        for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
        {
          FaceNode fn(cell.local_id, f, fnode);
          node_tracker_[fn] = AAHD_NodeIndex(0, true, true, false, false);
        }
      }
    }
  }
}

void
AAHD_FLUDSCommonData::CopyFlattenNodeIndexToDevice(const SpatialDiscretization& sdm)
{
  const MeshContinuum& grid = *(spds_.GetGrid());
  const std::size_t num_local_cells = grid.local_cells.size();
  const std::size_t num_angles = all_omegas_.empty() ? 1 : all_omegas_.size();

  crb::HostVector<std::uint64_t> cell_offset;
  crb::HostVector<std::uint64_t> primary_data;
  std::map<std::uint32_t, std::size_t> cell_offset_slot;
  std::uint64_t data_start = 2 * num_local_cells;
  std::uint64_t data_pos = data_start;
  for (const Cell& cell : grid.local_cells)
  {
    cell_offset_slot[cell.local_id] = cell_offset.size();
    cell_offset.push_back(data_pos);
    std::uint64_t num_nodes = 0;
    for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
    {
      std::uint32_t num_face_nodes = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
      for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
      {
        FaceNode node(cell.local_id, f, fnode);
        AAHD_NodeIndex index = node_tracker_.at(node);
        primary_data.push_back(index.GetCoreValue());
        ++num_nodes;
        ++data_pos;
      }
    }
    cell_offset.push_back(num_nodes);
  }

  const std::size_t header_size = cell_offset.size();
  const std::size_t data_size = primary_data.size();
  const std::size_t flat_size = header_size + data_size;

  DeallocateDeviceMemory();
  device_per_angle_indexes_.resize(num_angles, nullptr);

  {
    crb::DeviceMemory<std::uint64_t> device_mem_ptr(flat_size);
    crb::copy(device_mem_ptr, cell_offset, header_size);
    crb::copy(device_mem_ptr, primary_data, data_size, 0, header_size);
    device_per_angle_indexes_[0] = device_mem_ptr.release();
  }
  device_node_indexes_ = device_per_angle_indexes_[0];

  if (num_angles > 1)
  {
    std::size_t running_delayed = delayed_local_node_stack_size_;

    for (std::size_t a = 1; a < num_angles; ++a)
    {
      crb::HostVector<std::uint64_t> patched_data = primary_data;
      const Vector3& omega_a = all_omegas_[a];
      std::size_t additional_delayed = 0;

      for (const Cell& cell : grid.local_cells)
        for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
        {
          const CellFace& face = cell.faces[f];
          const FaceOrientation& orient = spds_.GetCellFaceOrientations()[cell.local_id][f];

          if (not face.has_neighbor)
          {
            const bool incoming = omega_a.Dot(face.normal) < 0.0;
            const auto [reflecting, angle_dependent] =
              boundary_node_flags_.at(FaceNode(cell.local_id, f, 0));

            std::uint64_t face_off = 0;
            for (std::uint32_t j = 0; j < f; ++j)
              face_off += sdm.GetCellMapping(cell).GetNumFaceNodes(j);

            const std::uint32_t num_fn = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
            for (std::uint32_t fn = 0; fn < num_fn; ++fn)
            {
              const std::uint64_t pos =
                cell_offset[cell_offset_slot.at(cell.local_id)] - data_start + face_off + fn;
              const AAHD_NodeIndex primary_index(primary_data[pos]);
              if (incoming)
                patched_data[pos] = AAHD_NodeIndex(primary_index.GetIndex(),
                                                   false,
                                                   true,
                                                   reflecting,
                                                   angle_dependent)
                                      .GetCoreValue();
              else if (reflecting)
                patched_data[pos] =
                  AAHD_NodeIndex(primary_index.GetIndex(), true, true, true, true).GetCoreValue();
              else
                patched_data[pos] = AAHD_NodeIndex(0, true, true, false, false).GetCoreValue();
            }
            continue;
          }

          if (orient != FaceOrientation::OUTGOING)
            continue;
          if (not face.IsNeighborLocal(&grid))
            continue;
          if (omega_a.Dot(face.normal) >= 0.0)
            continue;

          const std::uint32_t nbr_idx = face.GetNeighborLocalID(&grid);
          const FaceNodalMapping& fnm = grid_nodal_mappings_[cell.local_id][f];
          const std::uint32_t num_fn = sdm.GetCellMapping(cell).GetNumFaceNodes(f);

          std::uint64_t face_off = 0;
          for (std::uint32_t j = 0; j < f; ++j)
            face_off += sdm.GetCellMapping(cell).GetNumFaceNodes(j);

          const Cell& nbr_cell = grid.local_cells[nbr_idx];
          const std::uint32_t assoc_f = static_cast<std::uint32_t>(fnm.associated_face_);
          std::uint64_t nbr_face_off = 0;
          for (std::uint32_t j = 0; j < assoc_f; ++j)
            nbr_face_off += sdm.GetCellMapping(nbr_cell).GetNumFaceNodes(j);

          for (std::uint32_t fn = 0; fn < num_fn; ++fn)
          {
            const std::uint64_t del_idx = running_delayed + additional_delayed;
            const std::uint64_t pos =
              cell_offset[cell_offset_slot.at(cell.local_id)] - data_start + face_off + fn;
            const std::uint32_t mapped_fn =
              static_cast<std::uint32_t>(fnm.face_node_mapping_.at(fn));
            const std::uint64_t nbr_pos =
              cell_offset[cell_offset_slot.at(nbr_idx)] - data_start + nbr_face_off + mapped_fn;

            patched_data[pos] =
              AAHD_NodeIndex(del_idx, false, false, true, true).GetCoreValue();
            patched_data[nbr_pos] =
              AAHD_NodeIndex(del_idx, true, false, true, true).GetCoreValue();
            ++additional_delayed;
          }
        }

      for (const Cell& cell : grid.local_cells)
        for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
        {
          const CellFace& face = cell.faces[f];
          if (not face.has_neighbor or face.IsNeighborLocal(&grid))
            continue;
          const FaceOrientation& orient = spds_.GetCellFaceOrientations()[cell.local_id][f];
          const bool incoming_a = omega_a.Dot(face.normal) < 0.0;
          if ((orient == FaceOrientation::OUTGOING and !incoming_a) or
              (orient == FaceOrientation::INCOMING and incoming_a))
            continue;

          std::uint64_t face_off = 0;
          for (std::uint32_t j = 0; j < f; ++j)
            face_off += sdm.GetCellMapping(cell).GetNumFaceNodes(j);

          const std::uint32_t num_fn = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
          for (std::uint32_t fn = 0; fn < num_fn; ++fn)
          {
            const std::uint64_t pos =
              cell_offset[cell_offset_slot.at(cell.local_id)] - data_start + face_off + fn;
            const FaceNode node(cell.local_id, f, fn);
            if (incoming_a)
            {
              const auto idx = promoted_delayed_incoming_indices_.at({a, node}) +
                               nonlocal_delayed_incoming_node_offsets_.back();
              patched_data[pos] = AAHD_NodeIndex(idx, false, false, false, true).GetCoreValue();
            }
            else
            {
              const auto idx = promoted_delayed_outgoing_indices_.at({a, node});
              patched_data[pos] = AAHD_NodeIndex(idx, true, false, false, true).GetCoreValue();
            }
          }
        }

      running_delayed += additional_delayed;

      crb::DeviceMemory<std::uint64_t> device_mem_ptr(flat_size);
      crb::copy(device_mem_ptr, cell_offset, header_size);
      crb::copy(device_mem_ptr, patched_data, data_size, 0, header_size);
      device_per_angle_indexes_[a] = device_mem_ptr.release();
    }

    total_delayed_local_node_stack_size_ = running_delayed;
  }

  crb::HostVector<std::uint64_t*> host_ptrs;
  for (std::size_t a = 0; a < num_angles; ++a)
    host_ptrs.push_back(device_per_angle_indexes_[a]);
  crb::DeviceMemory<std::uint64_t*> dev_ptrs(num_angles);
  crb::copy(dev_ptrs, host_ptrs, num_angles);
  device_index_ptr_array_ = dev_ptrs.release();
}

void
AAHD_FLUDSCommonData::DeallocateDeviceMemory()
{
  for (auto& ptr : device_per_angle_indexes_)
  {
    if (ptr)
    {
      crb::DeviceMemory<std::uint64_t> device_mem_ptr(ptr);
      device_mem_ptr.reset();
      ptr = nullptr;
    }
  }
  device_per_angle_indexes_.clear();
  device_node_indexes_ = nullptr;

  if (device_index_ptr_array_)
  {
    crb::DeviceMemory<std::uint64_t*> device_mem_ptr(device_index_ptr_array_);
    device_mem_ptr.reset();
    device_index_ptr_array_ = nullptr;
  }
}

} // namespace opensn
