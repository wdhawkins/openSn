// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/reflecting_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/vacuum_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/isotropic_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/arbitrary_boundary.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/wgdsa.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/acceleration/tgdsa.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/caliper_scopes.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace opensn
{

namespace
{

std::set<std::uint64_t>
GetGlobalUniqueBoundaryIDs(const std::shared_ptr<MeshContinuum>& grid, mpi::Communicator& mpi_comm)
{
  std::set<std::uint64_t> local_unique_bids_set;
  for (const auto& cell : grid->local_cells)
    for (const auto& face : cell.faces)
      if (not face.has_neighbor)
        local_unique_bids_set.insert(face.neighbor_id);

  const std::vector<std::uint64_t> local_unique_bids(local_unique_bids_set.begin(),
                                                     local_unique_bids_set.end());

  std::vector<std::uint64_t> recvbuf;
  mpi_comm.all_gather(local_unique_bids, recvbuf);

  std::set<std::uint64_t> global_unique_bids_set = local_unique_bids_set;
  global_unique_bids_set.insert(recvbuf.begin(), recvbuf.end());

  return global_unique_bids_set;
}

} // namespace

const std::map<uint64_t, std::shared_ptr<SweepBoundary>>&
DiscreteOrdinatesProblem::GetSweepBoundaries() const
{
  return sweep_boundaries_;
}

const std::map<uint64_t, BoundaryDefinition>&
DiscreteOrdinatesProblem::GetBoundaryDefinitions() const
{
  return boundary_definitions_;
}

void
DiscreteOrdinatesProblem::SetBoundaryOptions(const std::vector<InputParameters>& boundary_params,
                                             bool clear_existing)
{
  if (clear_existing)
    boundary_definitions_.clear();

  for (const auto& params : boundary_params)
    UpdateBoundaryDefinition(params);

  if (clear_existing or not boundary_params.empty())
  {
    InitializeBoundaries();
    RebuildBoundaryRuntimeData();
  }
}

void
DiscreteOrdinatesProblem::UpdateBoundaryDefinition(const InputParameters& params)
{
  const auto boundary_name = params.GetParamValue<std::string>("name");
  const auto coord_sys = grid_->GetCoordinateSystem();
  const auto bnd_name_map = grid_->GetBoundaryNameMap();
  const auto mesh_type = grid_->GetType();

  // If we're using RZ, the user should use rmin/rmax/zmin/zmax and we'll
  // map internally to xmin/xmax/ymin/max
  std::string lookup_name = boundary_name;
  if (coord_sys == CoordinateSystemType::CYLINDRICAL and mesh_type == MeshType::ORTHOGONAL and
      grid_->GetDimension() == 2)
  {
    if (boundary_name != "rmin" and boundary_name != "rmax" and boundary_name != "zmin" and
        boundary_name != "zmax")
    {
      throw std::runtime_error(GetName() + ": Boundary name '" + boundary_name +
                               "' is invalid for cylindrical orthogonal meshes. "
                               "Use rmin, rmax, zmin, zmax.");
    }

    const std::map<std::string, std::string> rz_map = {
      {"rmin", "xmin"}, {"rmax", "xmax"}, {"zmin", "ymin"}, {"zmax", "ymax"}};
    const auto rz_it = rz_map.find(boundary_name);
    if (rz_it != rz_map.end())
      lookup_name = rz_it->second;
  }

  const auto it = bnd_name_map.find(lookup_name);
  if (it == bnd_name_map.end())
  {
    throw std::runtime_error("Boundary name \"" + boundary_name + "\" not found in mesh.");
  }
  const auto bid = it->second;
  boundary_definitions_[bid] = BoundaryDefinition(params, GetNumGroups());
}

void
DiscreteOrdinatesProblem::ClearBoundaries()
{
  boundary_definitions_.clear();
  InitializeBoundaries();
  RebuildBoundaryRuntimeData();
}

void
DiscreteOrdinatesProblem::RebuildBoundaryRuntimeData()
{
  if (boundary_runtime_data_initialized_)
    return;

  const bool solver_schemes_initialized =
    ags_solver_ or (not wgs_solvers_.empty()) or (not wgs_contexts_.empty());

  for (auto& [bid, boundary] : sweep_boundaries_)
    boundary->InitializeReflectingMap(groupsets_);
  if (use_gpus_)
    SortAngleSetsAngleIndices();
  for (auto& [bid, boundary] : sweep_boundaries_)
    boundary->InitializeAngleDependent(groupsets_);
  boundary_bank_.ShrinkToFit();
  boundary_bank_.DisableAllocation();
  InitializeBoundaryCarrier();
  if (sweep_type_ == "AAH" and use_gpus_)
    UpdateAAHD_FLUDSCommonDataWithBoundary();
  for (auto& groupset : groupsets_)
    groupset.angle_agg->SetupAngleSetDependencies();

  if (solver_schemes_initialized)
  {
    for (auto& groupset : groupsets_)
    {
      WGDSA::CleanUp(groupset);
      TGDSA::CleanUp(groupset);
      WGDSA::Init(*this, groupset);
      TGDSA::Init(*this, groupset);
    }
    ReinitializeSolverSchemes();
  }

  boundary_runtime_data_initialized_ = true;
}

Vector3
DiscreteOrdinatesProblem::ComputeReflectingBoundaryNormal(uint64_t bid) const
{
  const double EPSILON = 1.0e-12;
  std::unique_ptr<Vector3> n_ptr = nullptr;
  for (const auto& cell : grid_->local_cells)
  {
    for (const auto& face : cell.faces)
    {
      if (not face.has_neighbor and face.neighbor_id == bid)
      {
        if (not n_ptr)
          n_ptr = std::make_unique<Vector3>(face.normal);
        if (std::fabs(face.normal.Dot(*n_ptr) - 1.0) > EPSILON)
          throw std::logic_error(GetName() +
                                 ": Not all face normals are, within tolerance, locally the same "
                                 "for the reflecting boundary condition requested");
      }
    }
  }

  const int local_has_bid = n_ptr != nullptr ? 1 : 0;
  const Vector3 local_normal = local_has_bid ? *n_ptr : Vector3(0.0, 0.0, 0.0);

  std::vector<int> locJ_has_bid(opensn::mpi_comm.size(), 1);
  std::vector<double> locJ_n_val(opensn::mpi_comm.size() * 3L, 0.0);

  mpi_comm.all_gather(local_has_bid, locJ_has_bid);
  std::vector<double> lnv = {local_normal.x, local_normal.y, local_normal.z};
  mpi_comm.all_gather(lnv.data(), 3, locJ_n_val.data(), 3);

  Vector3 global_normal;
  for (int j = 0; j < opensn::mpi_comm.size(); ++j)
  {
    if (locJ_has_bid[j])
    {
      int offset = 3 * j;
      const double* n = &locJ_n_val[offset];
      const Vector3 locJ_normal(n[0], n[1], n[2]);

      if (local_has_bid)
        if (std::fabs(local_normal.Dot(locJ_normal) - 1.0) > EPSILON)
          throw std::logic_error(GetName() +
                                 ": Not all face normals are, within tolerance, globally the same "
                                 "for the reflecting boundary condition requested");

      global_normal = locJ_normal;
    }
  }

  return global_normal;
}

void
DiscreteOrdinatesProblem::InitializeBoundaries()
{
  CALI_CXX_MARK_SCOPE("Boundaries");

  ResetBoundaryCarrier();
  boundary_runtime_data_initialized_ = false;
  has_reflecting_boundaries_ = false;
  has_time_dependent_boundaries_ = false;

  ValidateBoundaryConfiguration();

  // Determine boundary-ids involved in the problem
  const auto unique_bids_set = GetGlobalUniqueBoundaryIDs(grid_, mpi_comm);
  for (const auto bid : unique_bids_set)
  {
    if (boundary_definitions_.find(bid) == boundary_definitions_.end())
      boundary_definitions_.emplace(bid, BoundaryDefinition());
  }
  boundary_bank_ = BoundaryBank(groupsets_);

  sweep_boundaries_.clear();
  const auto coord_sys = MapGeometryTypeToCoordSys(geometry_type_);
  for (auto bid : unique_bids_set)
  {
    auto& bndry_def = boundary_definitions_.find(bid)->second;

    switch (bndry_def.type)
    {
      case LBSBoundaryType::VACUUM:
      {
        sweep_boundaries_[bid] = std::make_shared<VacuumBoundary>(boundary_bank_);
        break;
      }
      case LBSBoundaryType::ISOTROPIC:
      {
        sweep_boundaries_[bid] = std::make_shared<IsotropicBoundary>(boundary_bank_,
                                                                     groupsets_,
                                                                     bndry_def.group_strength,
                                                                     bndry_def.start_time,
                                                                     bndry_def.end_time);
        has_time_dependent_boundaries_ = true;
        break;
      }
      case LBSBoundaryType::ARBITRARY:
      {
        sweep_boundaries_[bid] = std::make_shared<ArbitraryBoundary>(
          boundary_bank_, groupsets_, bndry_def.time_angular_flux_function);
        has_time_dependent_boundaries_ = true;
        break;
      }
      case LBSBoundaryType::REFLECTING:
      {
        const auto global_normal = ComputeReflectingBoundaryNormal(bid);
        sweep_boundaries_[bid] = std::make_shared<ReflectingBoundary>(
          boundary_bank_, bid, grid_, groupsets_, global_normal, coord_sys);
        has_reflecting_boundaries_ = true;
        break;
      }
      default:
      {
        throw std::logic_error("Boundary type not implemented.");
      }
    }
  }

  for (auto& [bid, bndry] : sweep_boundaries_)
  {
    bndry->SetOpposingReflected(bid, sweep_boundaries_);
  }
}

#ifndef __OPENSN_WITH_GPU__
void
DiscreteOrdinatesProblem::InitializeBoundaryCarrier()
{
}

void
DiscreteOrdinatesProblem::TransferDeviceBoundaryData(int groupset_id,
                                                     bool host_to_device,
                                                     bool force)
{
}

void
DiscreteOrdinatesProblem::ResetBoundaryCarrier()
{
}
#endif

} // namespace opensn
