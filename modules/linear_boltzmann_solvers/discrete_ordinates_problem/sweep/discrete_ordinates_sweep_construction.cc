// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_runtime_builder.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aah_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_csda_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_sweep_chunk_td.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk_td.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/utils/error.h"
#include "framework/utils/caliper_scopes.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace opensn
{

SweepKind
ParseSweepKind(const std::string& sweep_type, const std::string& problem_name)
{
  if (sweep_type == "AAH")
    return SweepKind::AAH;
  if (sweep_type == "CBC")
    return SweepKind::CBC;
  OpenSnInvalidArgument(problem_name + ": Unsupported sweep type \"" + sweep_type + "\"");
}

void
DiscreteOrdinatesProblem::InitializeSweepDataStructures()
{
  CALI_CXX_MARK_SCOPE("SweepDataStructures");

  auto sweep_runtime = BuildSweepRuntime(
    GetName(), groupsets_, grid_, sweep_type_, use_gpus_, *discretization_, grid_nodal_mappings_);

  quadrature_unq_so_grouping_map_ = std::move(sweep_runtime.quadrature_unq_so_grouping_map);
  quadrature_spds_map_ = std::move(sweep_runtime.quadrature_spds_map);
  quadrature_fluds_commondata_map_ = std::move(sweep_runtime.quadrature_fluds_commondata_map);

  // Prototype diagnostic: write each rank's sweep graph for offline critical-path modeling.
  // Per SPDS: level count, predecessor and delayed-predecessor locations, the levels of local
  // cells with outgoing faces to other ranks, and for each incoming face from another rank the
  // upwind cell's global id and rank with the level of the local cell that reads it.
  if (const char* prefix = std::getenv("OPENSN_SWEEP_GRAPH_DUMP"))
  {
    const int rank = opensn::mpi_comm.rank();
    std::ofstream out(std::string(prefix) + ".r" + std::to_string(rank) + ".bin", std::ios::binary);
    auto put = [&out](const auto& v) { out.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto put_vec = [&](const std::vector<int>& v)
    {
      put(static_cast<std::int32_t>(v.size()));
      for (int x : v)
        put(static_cast<std::int32_t>(x));
    };
    std::int32_t num_spds = 0;
    for (const auto& [quad, spds_list] : quadrature_spds_map_)
      num_spds += static_cast<std::int32_t>(spds_list.size());
    put(static_cast<std::int32_t>(rank));
    put(num_spds);
    for (const auto& [quad, spds_list] : quadrature_spds_map_)
      for (const auto& spds : spds_list)
      {
        const auto& levels = spds->GetLevelizedLocalSubgrid();
        std::vector<std::int32_t> level_of(grid_->GetLocalCells().size(), -1);
        for (std::size_t l = 0; l < levels.size(); ++l)
          for (auto c : levels[l])
            level_of[c] = static_cast<std::int32_t>(l);
        const auto& omega = spds->GetOmega();
        std::vector<std::pair<std::uint64_t, std::int32_t>> outgoing;
        struct Incoming
        {
          std::uint64_t up_gid;
          std::int32_t up_rank;
          std::int32_t level;
        };
        std::vector<Incoming> incoming;
        for (const auto& cell_ptr : grid_->GetLocalCells())
        {
          const auto& cell = *cell_ptr;
          bool has_outgoing = false;
          for (const auto& face : cell.faces)
          {
            if (not face.has_neighbor or grid_->IsCellLocal(face.neighbor_id))
              continue;
            const double mu = omega.Dot(face.normal);
            if (mu > 0.0)
              has_outgoing = true;
            else if (mu < 0.0)
              incoming.push_back(
                {face.neighbor_id,
                 static_cast<std::int32_t>(grid_->GetGlobalCell(face.neighbor_id).partition_id),
                 level_of[cell.local_id]});
          }
          if (has_outgoing)
            outgoing.emplace_back(cell.global_id, level_of[cell.local_id]);
        }
        put(static_cast<std::int32_t>(levels.size()));
        put_vec(spds->GetLocationDependencies());
        put_vec(spds->GetDelayedLocationDependencies());
        put(static_cast<std::int32_t>(outgoing.size()));
        for (const auto& [gid, level] : outgoing)
        {
          put(gid);
          put(level);
        }
        put(static_cast<std::int32_t>(incoming.size()));
        for (const auto& in : incoming)
        {
          put(in.up_gid);
          put(in.up_rank);
          put(in.level);
        }
      }
    log.Log() << "Wrote sweep graph dump " << prefix << ".r<rank>.bin";
  }
}

#ifndef __OPENSN_WITH_GPU__
std::shared_ptr<FLUDS>
DiscreteOrdinatesProblem::CreateAAHD_FLUDS(unsigned int num_groups,
                                           std::size_t num_angles,
                                           const FLUDSCommonData& common_data)
{
  throw std::runtime_error(
    "DiscreteOrdinatesProblem::CreateAAHD_FLUDS : OPENSN_WITH_CUDA not enabled.");
  return {};
}

std::shared_ptr<AngleSet>
DiscreteOrdinatesProblem::CreateAAHD_AngleSet(
  size_t id,
  const LBSGroupset& groupset,
  const SPDS& spds,
  std::shared_ptr<FLUDS>& fluds,
  std::vector<size_t>& angle_indices,
  std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
  int maximum_message_size,
  const SweepCommunicator& sweep_communicator)
{
  throw std::runtime_error(
    "DiscreteOrdinatesProblem::CreateAAHD_AngleSet : OPENSN_WITH_CUDA not enabled.");
  return {};
}

std::shared_ptr<SweepChunk>
DiscreteOrdinatesProblem::CreateAAHD_SweepChunk(LBSGroupset& groupset)
{
  throw std::runtime_error(
    "DiscreteOrdinatesProblem::CreateAAHD_SweepChunk : OPENSN_WITH_CUDA not enabled.");
  return {};
}

std::shared_ptr<FLUDS>
DiscreteOrdinatesProblem::CreateCBCD_FLUDS(std::size_t num_groups,
                                           std::size_t num_angles,
                                           std::size_t num_local_cells,
                                           const FLUDSCommonData& common_data,
                                           const UnknownManager& psi_uk_man,
                                           const SpatialDiscretization& sdm,
                                           bool save_angular_flux)
{
  throw std::runtime_error(
    "DiscreteOrdinatesProblem::CreateCBCD_FLUDS : OPENSN_WITH_CUDA not enabled.");
  return {};
}

std::shared_ptr<AngleSet>
DiscreteOrdinatesProblem::CreateCBCD_AngleSet(
  size_t id,
  const LBSGroupset& groupset,
  const SPDS& spds,
  std::shared_ptr<FLUDS>& fluds,
  std::vector<size_t>& angle_indices,
  std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
  const SweepCommunicator& sweep_communicator)
{
  throw std::runtime_error(
    "DiscreteOrdinatesProblem::CreateCBCD_AngleSet : OPENSN_WITH_CUDA not enabled.");
  return {};
}

std::shared_ptr<SweepChunk>
DiscreteOrdinatesProblem::CreateCBCDSweepChunk(LBSGroupset& groupset)
{
  throw std::runtime_error(
    "DiscreteOrdinatesProblem::CreateCBCDSweepChunk : OPENSN_WITH_CUDA not enabled.");
  return {};
}
#endif

void
DiscreteOrdinatesProblem::InitFluxDataStructures(LBSGroupset& groupset)
{
  CALI_CXX_MARK_SCOPE("FluxDataStructures");

  const auto& quadrature_sweep_info = quadrature_unq_so_grouping_map_[groupset.quadrature];

  const auto& unique_so_groupings = quadrature_sweep_info.first;
  const auto& dir_id_to_so_map = quadrature_sweep_info.second;

  const size_t gs_num_grps = groupset.GetNumGroups();

  // Passing the sweep boundaries to the angle aggregation
  groupset.angle_agg = std::make_shared<AngleAggregation>(
    groupset, sweep_boundaries_, groupset.quadrature, grid_, options_.csda_enabled);

  size_t angle_set_id = 0;
  for (const auto& so_grouping : unique_so_groupings)
  {
    const size_t master_dir_id = so_grouping.front();
    const size_t so_id = dir_id_to_so_map.at(master_dir_id);

    const auto& sweep_ordering = quadrature_spds_map_[groupset.quadrature][so_id];
    const auto& fluds_common_data = *quadrature_fluds_commondata_map_[groupset.quadrature][so_id];

    std::vector<size_t> angle_indices(so_grouping.begin(), so_grouping.end());

    switch (ParseSweepKind(sweep_type_, GetName()))
    {
      case SweepKind::AAH:
      {
        std::shared_ptr<FLUDS> fluds;
        if (use_gpus_)
        {
          fluds = CreateAAHD_FLUDS(gs_num_grps, angle_indices.size(), fluds_common_data);
        }
        else
        {
          fluds =
            std::make_shared<AAH_FLUDS>(gs_num_grps,
                                        angle_indices.size(),
                                        dynamic_cast<const AAH_FLUDSCommonData&>(fluds_common_data),
                                        options_.csda_enabled);
        }

        std::shared_ptr<AngleSet> angle_set;
        if (use_gpus_)
        {
          angle_set = CreateAAHD_AngleSet(angle_set_id++,
                                          groupset,
                                          *sweep_ordering,
                                          fluds,
                                          angle_indices,
                                          sweep_boundaries_,
                                          options_.max_mpi_message_size,
                                          *sweep_communicator_);
        }
        else
        {
          angle_set = std::make_shared<AAH_AngleSet>(angle_set_id++,
                                                     groupset,
                                                     *sweep_ordering,
                                                     fluds,
                                                     angle_indices,
                                                     sweep_boundaries_,
                                                     options_.max_mpi_message_size,
                                                     options_.csda_enabled,
                                                     *sweep_communicator_);
        }
        groupset.angle_agg->GetAngleSetGroups().push_back(angle_set);
        break;
      }
      case SweepKind::CBC:
      {
        std::shared_ptr<FLUDS> fluds;
        if (use_gpus_)
        {
          fluds = CreateCBCD_FLUDS(gs_num_grps,
                                   angle_indices.size(),
                                   grid_->GetLocalCellCount(),
                                   fluds_common_data,
                                   groupset.psi_uk_man_,
                                   *discretization_,
                                   (not GetPsiNewLocal()[groupset.id].empty()));
        }
        else
        {
          fluds =
            std::make_shared<CBC_FLUDS>(gs_num_grps,
                                        angle_indices.size(),
                                        dynamic_cast<const CBC_FLUDSCommonData&>(fluds_common_data),
                                        groupset.psi_uk_man_,
                                        *discretization_);
        }

        std::shared_ptr<AngleSet> angle_set;
        if (use_gpus_)
        {
          angle_set = CreateCBCD_AngleSet(angle_set_id++,
                                          groupset,
                                          *sweep_ordering,
                                          fluds,
                                          angle_indices,
                                          sweep_boundaries_,
                                          *sweep_communicator_);
        }
        else
        {
          angle_set = std::make_shared<CBC_AngleSet>(angle_set_id++,
                                                     groupset,
                                                     *sweep_ordering,
                                                     fluds,
                                                     angle_indices,
                                                     sweep_boundaries_,
                                                     *sweep_communicator_);
        }

        groupset.angle_agg->GetAngleSetGroups().push_back(angle_set);
        break;
      }
    }
  } // for so_grouping

  groupset.angle_agg->BuildDirnumToAnglesetMap();

  opensn::mpi_comm.barrier();
}

std::shared_ptr<SweepChunk>
DiscreteOrdinatesProblem::SetSweepChunk(LBSGroupset& groupset)
{

  const auto mode = sweep_chunk_mode_.value_or(SweepChunkMode::DEFAULT);

  const bool use_time_dependent_chunk = (mode == SweepChunkMode::TIME_DEPENDENT);

  switch (ParseSweepKind(sweep_type_, GetName()))
  {
    case SweepKind::AAH:
      if (options_.csda_enabled)
        return std::make_shared<AAHCSDASweepChunk>(*this, groupset);
      if (use_time_dependent_chunk)
        return std::make_shared<AAHSweepChunkTD>(*this, groupset);
      if (use_gpus_)
        return CreateAAHD_SweepChunk(groupset);
      return std::make_shared<AAHSweepChunk>(*this, groupset);
    case SweepKind::CBC:
      if (use_time_dependent_chunk)
        return std::make_shared<CBCSweepChunkTD>(*this, groupset);
      if (use_gpus_)
        return CreateCBCDSweepChunk(groupset);
      return std::make_shared<CBCSweepChunk>(*this, groupset);
  }
  OpenSnLogicalError("Unreachable: unknown SweepKind.");
}

} // namespace opensn
