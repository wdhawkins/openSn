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
  // initialize flatten index and cell offset
  crb::HostVector<std::uint64_t> cell_offset;
  crb::HostVector<std::uint64_t> data;
  // loop for each cell
  const MeshContinuum& grid = *(spds_.GetGrid());
  std::uint64_t offset = 2 * grid.local_cells.size();
  for (const Cell& cell : grid.local_cells)
  {
    // record the offset
    cell_offset.push_back(offset);
    // record each face node to the data
    std::uint64_t num_nodes = 0;
    for (std::uint32_t f = 0; f < cell.faces.size(); ++f)
    {
      std::uint32_t num_face_nodes = sdm.GetCellMapping(cell).GetNumFaceNodes(f);
      for (std::uint32_t fnode = 0; fnode < num_face_nodes; ++fnode)
      {
        FaceNode node(cell.local_id, f, fnode);
        AAHD_NodeIndex index = node_tracker_.at(node);
        data.push_back(index.GetCoreValue());
        ++num_nodes;
        ++offset;
      }
    }
    cell_offset.push_back(num_nodes);
  }
  // copy data to GPU
  DeallocateDeviceMemory();
  crb::DeviceMemory<std::uint64_t> device_mem_ptr(cell_offset.size() + data.size());
  crb::copy(device_mem_ptr, cell_offset, cell_offset.size());
  crb::copy(device_mem_ptr, data, data.size(), 0, cell_offset.size());
  device_node_indexes_ = device_mem_ptr.release();

  // Build level-traversal-ordered face node data for cache-friendly kernel access.
  // Within each level, cells are stored in sweep-dependency order, so sequential reads
  // of level_cell_starts[level_abs_start + cell_idx] hit the same cache lines for
  // the 100+ cells processed by a level kernel.
  const auto& topo_levels = spds_.GetLevelizedLocalSubgrid();
  crb::HostVector<std::uint64_t> level_node_data_host;
  crb::HostVector<std::uint32_t> level_cell_starts_host;
  level_abs_starts_.clear();
  std::uint32_t abs_cell = 0;
  for (const auto& level : topo_levels)
  {
    level_abs_starts_.push_back(abs_cell);
    for (const std::uint32_t cell_local_idx : level)
    {
      level_cell_starts_host.push_back(static_cast<std::uint32_t>(level_node_data_host.size()));
      const Cell& cell2 = grid.local_cells[cell_local_idx];
      for (std::uint32_t f = 0; f < cell2.faces.size(); ++f)
      {
        const std::uint32_t nfn = sdm.GetCellMapping(cell2).GetNumFaceNodes(f);
        for (std::uint32_t fnode = 0; fnode < nfn; ++fnode)
        {
          FaceNode fn(cell_local_idx, f, fnode);
          level_node_data_host.push_back(node_tracker_.at(fn).GetCoreValue());
        }
      }
      ++abs_cell;
    }
  }
  level_abs_starts_.push_back(abs_cell);
  level_cell_starts_host.push_back(static_cast<std::uint32_t>(level_node_data_host.size()));

  if (!level_node_data_host.empty())
  {
    crb::DeviceMemory<std::uint64_t> lnd(level_node_data_host.size());
    crb::copy(lnd, level_node_data_host, level_node_data_host.size());
    device_level_node_data_ = lnd.release();
  }
  if (!level_cell_starts_host.empty())
  {
    crb::DeviceMemory<std::uint32_t> lcs(level_cell_starts_host.size());
    crb::copy(lcs, level_cell_starts_host, level_cell_starts_host.size());
    device_level_cell_starts_ = lcs.release();
  }
}

void
AAHD_FLUDSCommonData::DeallocateDeviceMemory()
{
  if (device_node_indexes_)
  {
    crb::DeviceMemory<std::uint64_t> device_mem_ptr(device_node_indexes_);
    device_mem_ptr.reset();
    device_node_indexes_ = nullptr;
  }
  if (device_level_node_data_)
  {
    crb::DeviceMemory<std::uint64_t> p(device_level_node_data_);
    p.reset();
    device_level_node_data_ = nullptr;
  }
  if (device_level_cell_starts_)
  {
    crb::DeviceMemory<std::uint32_t> p(device_level_cell_starts_);
    p.reset();
    device_level_cell_starts_ = nullptr;
  }
}

} // namespace opensn
