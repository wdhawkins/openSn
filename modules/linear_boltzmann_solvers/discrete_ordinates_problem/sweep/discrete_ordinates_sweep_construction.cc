// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep_runtime_builder.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/cbc_fluds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aah_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/cbc_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_sweep_chunk_td.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbc_sweep_chunk_td.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "framework/utils/error.h"
#include "framework/utils/caliper_scopes.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <list>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace opensn
{

namespace
{

void
RecursiveAngleSort(AngleSet* angleset,
                   AngleAggregation& angle_agg,
                   const std::vector<std::vector<std::uint32_t>>& reflected_maps,
                   std::set<AngleSet*>& unsorted,
                   std::set<AngleSet*>& sorted)
{
  sorted.insert(angleset);
  std::uint32_t angle_zero = angleset->GetAngleIndices()[0];
  for (const auto& reflected_map : reflected_maps)
  {
    auto reflected_angle_zero = reflected_map[angle_zero];
    auto* reflected_angleset = angle_agg.GetAngleSetForAngleIndex(reflected_angle_zero);
    if (sorted.contains(reflected_angleset))
    {
      bool is_coherent = std::equal(angleset->GetAngleIndices().begin(),
                                    angleset->GetAngleIndices().end(),
                                    reflected_angleset->GetAngleIndices().begin(),
                                    [&](std::uint32_t angle, std::uint32_t reflected)
                                    { return reflected_map[angle] == reflected; });
      if (not is_coherent)
        throw std::logic_error("Cannot find an unanimous sort order for the angle set.\n");
    }
    else
    {
      std::transform(angleset->GetAngleIndices().begin(),
                     angleset->GetAngleIndices().end(),
                     reflected_angleset->GetAngleIndices().begin(),
                     [&](std::uint32_t angle) { return reflected_map[angle]; });
      reflected_angleset->SyncDeviceAngleIndices();
      unsorted.erase(reflected_angleset);
      RecursiveAngleSort(reflected_angleset, angle_agg, reflected_maps, unsorted, sorted);
    }
  }
}

} // namespace

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
}

#ifndef __OPENSN_WITH_GPU__
void
DiscreteOrdinatesProblem::UpdateAAHD_FLUDSCommonDataWithBoundary()
{
  throw std::runtime_error("DiscreteOrdinatesProblem::UpdateAAHD_FLUDSCommonDataWithBoundary : "
                           "OPENSN_WITH_CUDA not enabled.");
}

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
  const MPICommunicatorSet& in_comm_set)
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
  const MPICommunicatorSet& in_comm_set)
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
  groupset.angle_agg =
    std::make_shared<AngleAggregation>(groupset, sweep_boundaries_, groupset.quadrature, grid_);

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
          fluds = std::make_shared<AAH_FLUDS>(
            gs_num_grps,
            angle_indices.size(),
            dynamic_cast<const AAH_FLUDSCommonData&>(fluds_common_data));
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
                                          *grid_local_comm_set_);
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
                                                     *grid_local_comm_set_);
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
                                   grid_->local_cells.size(),
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
                                          *grid_local_comm_set_);
        }
        else
        {
          angle_set = std::make_shared<CBC_AngleSet>(angle_set_id++,
                                                     groupset,
                                                     *sweep_ordering,
                                                     fluds,
                                                     angle_indices,
                                                     sweep_boundaries_,
                                                     *grid_local_comm_set_);
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

void
DiscreteOrdinatesProblem::SortAngleSetsAngleIndices()
{
  for (auto& groupset : groupsets_)
  {
    // build the total reflected map
    std::vector<std::vector<std::uint32_t>> reflected_maps;
    for (auto& [bid, boundary] : sweep_boundaries_)
      boundary->GetReflectedMap(groupset.id, reflected_maps);

    // check for each map and angle set that the mapped angle indices is also another angleset
    auto& angle_agg = *(groupset.angle_agg);
    std::list<std::set<std::uint32_t>> angle_indices_list;
    for (const auto& angleset : angle_agg)
    {
      const auto& angle_indices = angleset->GetAngleIndices();
      angle_indices_list.emplace_back(angle_indices.begin(), angle_indices.end());
    }
    for (const auto& reflected_map : reflected_maps)
    {
      std::list<std::set<std::uint32_t>> angle_indices_list_copy(angle_indices_list);
      for (auto it_in = angle_indices_list_copy.begin(); it_in != angle_indices_list_copy.end();)
      {
        std::set<std::uint32_t> reflected;
        std::transform(it_in->begin(),
                       it_in->end(),
                       std::inserter(reflected, reflected.end()),
                       [&](std::uint32_t n) { return reflected_map[n]; });
        auto it_out =
          std::find(angle_indices_list_copy.begin(), angle_indices_list_copy.end(), reflected);
        if (it_out == it_in or it_out == angle_indices_list_copy.end())
          throw std::logic_error("Angleset parity was broken for one of the reflected boundaries.");
        angle_indices_list_copy.erase(it_out);
        it_in = angle_indices_list_copy.erase(it_in);
      }
    }
    angle_indices_list.clear();

    // sort angle indices in anglesets
    std::set<AngleSet*> sorted_anglesets, unsorted_anglesets;
    std::transform(angle_agg.begin(),
                   angle_agg.end(),
                   std::inserter(unsorted_anglesets, unsorted_anglesets.end()),
                   [](auto& angleset) { return angleset.get(); });
    while (not unsorted_anglesets.empty())
    {
      AngleSet* angleset = *unsorted_anglesets.begin();
      unsorted_anglesets.erase(unsorted_anglesets.begin());
      RecursiveAngleSort(angleset, angle_agg, reflected_maps, unsorted_anglesets, sorted_anglesets);
    }
  }
}

} // namespace opensn
