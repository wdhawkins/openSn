// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/material_source_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"
#include "framework/utils/caliper_scopes.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

namespace opensn
{

namespace
{

/// Most (node, moment) outputs one material-source thread accumulates at a time.
constexpr std::uint32_t max_outputs_per_thread = 8;

struct MaterialSourceArguments
{
  const MaterialSourceCarrier::NodeData* nodes;
  const std::uint32_t* cell_node_offsets;
  const std::uint64_t* scatter_row_offsets;
  const std::uint32_t* scatter_columns;
  const double* scatter_values;
  const double* production;
  const std::uint8_t* fissionable;
  const std::uint32_t* ell_values;
  const std::uint32_t* ell_moment_offsets;
  const std::uint32_t* ell_moments;
  /// Flux moments in the problem layout, or only the isotropic moment node by node.
  const double* phi;
  double* q;
  std::uint64_t num_work_items;
  std::uint32_t num_ells;
  std::uint32_t num_groups;
  std::uint32_t num_moments;
  std::uint32_t groupset_first;
  std::uint32_t groupset_size;
  bool phi_isotropic_only;
  bool apply_wgs_scatter;
  bool apply_ags_scatter;
  bool apply_wgs_fission;
  bool apply_ags_fission;
  bool suppress_wg_scatter;
};

/**
 * One thread per (cell, harmonic degree, group) computes the source of that group for every node
 * of the cell and every moment of that degree, so each matrix entry it loads serves several
 * outputs. Each output sums its terms in the same order as SourceFunction.
 */
__CRB_GLOBAL_FUNC__ void
MaterialSourceKernel(MaterialSourceArguments args)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  const std::uint64_t work_item = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  const auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const std::uint64_t work_item = work_index.get_global_id(2);
#endif
  if (work_item >= args.num_work_items)
    return;

  const std::uint32_t group = args.groupset_first + work_item % args.groupset_size;
  const std::uint64_t cell_ell = work_item / args.groupset_size;
  const std::uint32_t ell_index = cell_ell % args.num_ells;
  const std::uint64_t cell = cell_ell / args.num_ells;
  const std::uint32_t ell = args.ell_values[ell_index];
  const std::uint32_t node_begin = args.cell_node_offsets[cell];
  const std::uint32_t num_cell_nodes = args.cell_node_offsets[cell + 1] - node_begin;
  const std::uint32_t moment_begin = args.ell_moment_offsets[ell_index];
  const std::uint32_t num_ell_moments = args.ell_moment_offsets[ell_index + 1] - moment_begin;
  const std::uint32_t material = args.nodes[node_begin].material_index;

  const bool scatter = args.apply_wgs_scatter or args.apply_ags_scatter;
  // Fission contributes only to the isotropic moment.
  const bool fission = ell == 0 and args.fissionable[material] != 0 and
                       (args.apply_wgs_fission or args.apply_ags_fission);
  if (not scatter and not fission)
    return;

  const std::uint32_t groupset_end = args.groupset_first + args.groupset_size;
  const std::uint64_t row =
    (static_cast<std::uint64_t>(material) * args.num_moments + ell) * (args.num_groups + 1) + group;
  const std::uint64_t production_row =
    (static_cast<std::uint64_t>(material) * args.num_groups + group) * args.num_groups;
  const std::uint32_t num_outputs = num_cell_nodes * num_ell_moments;

  for (std::uint32_t first_output = 0; first_output < num_outputs;
       first_output += max_outputs_per_thread)
  {
    const std::uint32_t count = num_outputs - first_output < max_outputs_per_thread
                                  ? num_outputs - first_output
                                  : max_outputs_per_thread;
    std::uint64_t q_address[max_outputs_per_thread];
    std::uint64_t phi_address[max_outputs_per_thread];
    double rhs[max_outputs_per_thread];
#pragma unroll
    for (std::uint32_t k = 0; k < max_outputs_per_thread; ++k)
    {
      rhs[k] = 0.0;
      q_address[k] = 0;
      phi_address[k] = 0;
      if (k < count)
      {
        const std::uint32_t output = first_output + k;
        const std::uint32_t node_index = node_begin + output / num_ell_moments;
        const std::uint32_t moment = args.ell_moments[moment_begin + output % num_ell_moments];
        q_address[k] =
          args.nodes[node_index].phi_address + static_cast<std::uint64_t>(moment) * args.num_groups;
        phi_address[k] = args.phi_isotropic_only
                           ? static_cast<std::uint64_t>(node_index) * args.num_groups
                           : q_address[k];
      }
    }

    if (args.apply_ags_scatter)
      for (std::uint64_t entry = args.scatter_row_offsets[row];
           entry < args.scatter_row_offsets[row + 1];
           ++entry)
      {
        const std::uint32_t source_group = args.scatter_columns[entry];
        if (source_group < args.groupset_first or source_group >= groupset_end)
        {
          const double value = args.scatter_values[entry];
#pragma unroll
          for (std::uint32_t k = 0; k < max_outputs_per_thread; ++k)
            if (k < count)
              rhs[k] += value * args.phi[phi_address[k] + source_group];
        }
      }
    if (args.apply_wgs_scatter)
      for (std::uint64_t entry = args.scatter_row_offsets[row];
           entry < args.scatter_row_offsets[row + 1];
           ++entry)
      {
        const std::uint32_t source_group = args.scatter_columns[entry];
        if (source_group >= args.groupset_first and source_group < groupset_end and
            not(args.suppress_wg_scatter and source_group == group))
        {
          const double value = args.scatter_values[entry];
#pragma unroll
          for (std::uint32_t k = 0; k < max_outputs_per_thread; ++k)
            if (k < count)
              rhs[k] += value * args.phi[phi_address[k] + source_group];
        }
      }

    if (fission)
    {
      if (args.apply_ags_fission)
        for (std::uint32_t source_group = 0; source_group < args.num_groups; ++source_group)
          if (source_group < args.groupset_first or source_group >= groupset_end)
          {
            const double value = args.production[production_row + source_group];
#pragma unroll
            for (std::uint32_t k = 0; k < max_outputs_per_thread; ++k)
              if (k < count)
                rhs[k] += value * args.phi[phi_address[k] + source_group];
          }
      if (args.apply_wgs_fission)
        for (std::uint32_t source_group = args.groupset_first; source_group < groupset_end;
             ++source_group)
        {
          const double value = args.production[production_row + source_group];
#pragma unroll
          for (std::uint32_t k = 0; k < max_outputs_per_thread; ++k)
            if (k < count)
              rhs[k] += value * args.phi[phi_address[k] + source_group];
        }
    }

#pragma unroll
    for (std::uint32_t k = 0; k < max_outputs_per_thread; ++k)
      if (k < count)
        args.q[q_address[k] + group] += rhs[k];
  }
}

struct FissionProductionArguments
{
  const MaterialSourceCarrier::NodeData* nodes;
  const double* production_column_sums;
  const std::uint8_t* fissionable;
  /// Isotropic flux moment node by node.
  const double* phi_isotropic;
  double* node_production;
  std::uint64_t num_nodes;
  std::uint32_t num_groups;
};

__CRB_GLOBAL_FUNC__ void
FissionProductionKernel(FissionProductionArguments args)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  const std::uint64_t node_index =
    static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  const auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const std::uint64_t node_index = work_index.get_global_id(2);
#endif
  if (node_index >= args.num_nodes)
    return;

  const auto node = args.nodes[node_index];
  double production = 0.0;
  if (args.fissionable[node.material_index] != 0)
  {
    // sum_g sum_g' F[g][g'] phi[g'] = sum_g' (sum_g F[g][g']) phi[g']
    const double* column_sums = args.production_column_sums +
                                static_cast<std::uint64_t>(node.material_index) * args.num_groups;
    const double* phi = args.phi_isotropic + node_index * args.num_groups;
    for (std::uint32_t source_group = 0; source_group < args.num_groups; ++source_group)
      production += column_sums[source_group] * phi[source_group];
    production *= node.volume_weight;
  }
  args.node_production[node_index] = production;
}

/// Returns a one-dimensional grid with one thread per work item. The threads of a launch dimension
/// must fit in 32 bits: HIP limits them so, and so does the Dim3 product that forms SYCL ranges.
crb::Dim3
LaunchGrid(const std::uint64_t num_work_items,
           const std::uint32_t block_size,
           const std::string& operation)
{
  const std::uint64_t num_blocks = (num_work_items + block_size - 1) / block_size;
  if (num_blocks * block_size > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error(operation + " launch grid is too large.");
  return crb::Dim3(static_cast<std::uint32_t>(num_blocks));
}

template <typename T>
crb::DeviceMemory<T>
MakeDeviceCopy(const std::vector<T>& values)
{
  crb::DeviceMemory<T> device(std::max<std::size_t>(values.size(), 1));
  if (not values.empty())
    crb::copy(device, values, values.size());
  return device;
}

} // namespace

MaterialSourceCarrier::MaterialSourceCarrier(const LBSProblem& lbs_problem)
  : num_groups_(lbs_problem.GetNumGroups()),
    num_moments_(lbs_problem.GetNumMoments()),
    vector_size_(lbs_problem.GetPhiNewLocal().size())
{
  if (num_groups_ > std::numeric_limits<std::uint32_t>::max() or
      num_moments_ > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("Too many groups or moments for GPU material-source assembly.");
  const auto& xs_map = lbs_problem.GetBlockID2XSMap();
  if (xs_map.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("Too many materials for GPU material-source assembly.");
  if (lbs_problem.GetLocalNodeCount() > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("Too many nodes for GPU material-source assembly.");

  std::map<unsigned int, std::uint32_t> block_to_material;
  std::uint32_t material_index = 0;
  for (const auto& [block_id, unused] : xs_map)
  {
    static_cast<void>(unused);
    block_to_material.emplace(block_id, material_index++);
  }

  std::vector<NodeData> nodes;
  nodes.reserve(lbs_problem.GetLocalNodeCount());
  std::vector<std::uint32_t> cell_node_offsets(1, 0);
  const auto& transport_views = lbs_problem.GetCellTransportViews();
  const auto& unit_cell_matrices = lbs_problem.GetUnitCellMatrices();
  for (const auto& cell : lbs_problem.GetGrid()->GetLocalCells())
  {
    const auto material = block_to_material.at(cell->block_id);
    const auto& transport_view = transport_views[cell->local_id];
    const auto& cell_matrices = unit_cell_matrices[cell->local_id];
    for (int node = 0; node < transport_view.GetNumNodes(); ++node)
      nodes.push_back(
        {transport_view.MapDOF(node, 0, 0), material, cell_matrices.intV_shapeI(node)});
    cell_node_offsets.push_back(static_cast<std::uint32_t>(nodes.size()));
  }
  num_nodes_ = nodes.size();
  num_cells_ = cell_node_offsets.size() - 1;
  node_phi_addresses_.resize(num_nodes_);
  std::transform(nodes.begin(),
                 nodes.end(),
                 node_phi_addresses_.begin(),
                 [](const NodeData& node) { return node.phi_address; });

  const std::size_t rows_per_material = num_moments_ * (num_groups_ + 1);
  std::vector<std::uint64_t> scatter_row_offsets(xs_map.size() * rows_per_material, 0);
  std::vector<std::uint32_t> scatter_columns;
  std::vector<double> scatter_values;
  std::vector<double> production(xs_map.size() * num_groups_ * num_groups_, 0.0);
  std::vector<std::uint8_t> fissionable(xs_map.size(), 0);
  const bool adjoint = lbs_problem.GetOptions().adjoint;

  material_index = 0;
  for (const auto& [block_id, xs] : xs_map)
  {
    static_cast<void>(block_id);
    const auto& transfer_matrices = xs->GetTransferMatrices(adjoint);
    for (std::size_t ell = 0; ell < num_moments_; ++ell)
    {
      const std::size_t row_base =
        (static_cast<std::size_t>(material_index) * num_moments_ + ell) * (num_groups_ + 1);
      for (std::size_t group = 0; group < num_groups_; ++group)
      {
        scatter_row_offsets[row_base + group] = scatter_values.size();
        if (ell < transfer_matrices.size())
          for (const auto& [unused_row, source_group, value] : transfer_matrices[ell].Row(group))
          {
            static_cast<void>(unused_row);
            scatter_columns.push_back(static_cast<std::uint32_t>(source_group));
            scatter_values.push_back(value);
          }
      }
      scatter_row_offsets[row_base + num_groups_] = scatter_values.size();
    }

    fissionable[material_index] = xs->IsFissionable() ? 1 : 0;
    if (xs->IsFissionable())
    {
      const auto& matrix = xs->GetProductionMatrix(adjoint);
      for (std::size_t group = 0; group < num_groups_; ++group)
        std::copy(matrix.at(group).begin(),
                  matrix.at(group).end(),
                  production.begin() +
                    (static_cast<std::size_t>(material_index) * num_groups_ + group) * num_groups_);
    }
    ++material_index;
  }

  std::vector<double> production_column_sums(xs_map.size() * num_groups_, 0.0);
  for (std::size_t material = 0; material < xs_map.size(); ++material)
    for (std::size_t group = 0; group < num_groups_; ++group)
      for (std::size_t source_group = 0; source_group < num_groups_; ++source_group)
        production_column_sums[material * num_groups_ + source_group] +=
          production[(material * num_groups_ + group) * num_groups_ + source_group];

  nodes_ = MakeDeviceCopy(nodes);
  cell_node_offsets_ = MakeDeviceCopy(cell_node_offsets);
  scatter_row_offsets_ = MakeDeviceCopy(scatter_row_offsets);
  scatter_columns_ = MakeDeviceCopy(scatter_columns);
  scatter_values_ = MakeDeviceCopy(scatter_values);
  production_ = MakeDeviceCopy(production);
  production_column_sums_ = MakeDeviceCopy(production_column_sums);
  fissionable_ = MakeDeviceCopy(fissionable);
  ell_values_ = crb::DeviceMemory<std::uint32_t>(std::max<std::size_t>(num_moments_, 1));
  ell_moment_offsets_ = crb::DeviceMemory<std::uint32_t>(num_moments_ + 1);
  ell_moments_ = crb::DeviceMemory<std::uint32_t>(std::max<std::size_t>(num_moments_, 1));
  phi_ = crb::DeviceMemory<double>(std::max<std::size_t>(vector_size_, 1));
  phi_isotropic_host_.resize(num_nodes_ * num_groups_);
  phi_isotropic_ = crb::DeviceMemory<double>(std::max<std::size_t>(phi_isotropic_host_.size(), 1));
  node_production_ = crb::DeviceMemory<double>(std::max<std::size_t>(num_nodes_, 1));
  node_production_host_.resize(num_nodes_);
}

void
MaterialSourceCarrier::UploadIsotropicFlux(const std::vector<double>& phi)
{
  CaliperRegionScope transfer_scope("GPUTransfer", CaliperGPUTransferScopeDepth());
  for (std::size_t node = 0; node < num_nodes_; ++node)
    std::copy_n(phi.data() + node_phi_addresses_[node],
                num_groups_,
                phi_isotropic_host_.data() + node * num_groups_);
  crb::copy(phi_isotropic_, phi_isotropic_host_, phi_isotropic_host_.size());
}

void
MaterialSourceCarrier::Apply(const LBSGroupset& groupset,
                             const std::vector<double>& q,
                             const std::vector<double>& phi,
                             double* q_device,
                             const SourceFlags source_flags)
{
  // A groupset that spans all groups has no across-groupset sources.
  const bool across_groupsets = groupset.GetNumGroups() < num_groups_;
  const bool apply_wgs_scatter = source_flags & APPLY_WGS_SCATTER_SOURCES;
  const bool apply_ags_scatter = across_groupsets and (source_flags & APPLY_AGS_SCATTER_SOURCES);
  const bool apply_wgs_fission = source_flags & APPLY_WGS_FISSION_SOURCES;
  const bool apply_ags_fission = across_groupsets and (source_flags & APPLY_AGS_FISSION_SOURCES);
  const bool suppress_wg_scatter = source_flags & SUPPRESS_WG_SCATTER;
  if (not apply_wgs_scatter and not apply_ags_scatter and not apply_wgs_fission and
      not apply_ags_fission)
    return;
  if (q.size() != vector_size_ or phi.size() != vector_size_)
    throw std::invalid_argument("GPU material-source vector size does not match the LBS problem.");
  if (q_device == nullptr)
    throw std::invalid_argument("GPU material-source destination pointer is null.");

  const auto& moment_map = groupset.quadrature->GetMomentToHarmonicsIndexMap();
  if (moment_map.size() != num_moments_)
    throw std::invalid_argument(
      "GPU material-source quadrature moment count does not match the LBS problem.");
  // Without scattering only the isotropic moment, harmonic degree 0, has a source.
  const bool isotropic_only = not apply_wgs_scatter and not apply_ags_scatter;
  std::map<std::uint32_t, std::vector<std::uint32_t>> moments_by_ell;
  for (std::uint32_t moment = 0; moment < num_moments_; ++moment)
  {
    const auto ell = static_cast<std::uint32_t>(moment_map[moment].ell);
    if (ell >= num_moments_)
      throw std::invalid_argument("GPU material-source harmonic degree is out of range.");
    if (ell == 0 or not isotropic_only)
      moments_by_ell[ell].push_back(moment);
  }
  std::vector<std::uint32_t> ell_values;
  std::vector<std::uint32_t> ell_moment_offsets(1, 0);
  std::vector<std::uint32_t> ell_moments;
  for (const auto& [ell, moments] : moments_by_ell)
  {
    ell_values.push_back(ell);
    ell_moments.insert(ell_moments.end(), moments.begin(), moments.end());
    ell_moment_offsets.push_back(static_cast<std::uint32_t>(ell_moments.size()));
  }

  {
    CaliperRegionScope transfer_scope("GPUTransfer", CaliperGPUTransferScopeDepth());
    crb::copy(ell_values_, ell_values, ell_values.size());
    crb::copy(ell_moment_offsets_, ell_moment_offsets, ell_moment_offsets.size());
    crb::copy(ell_moments_, ell_moments, ell_moments.size());
    if (not isotropic_only)
      crb::copy(phi_, phi, phi.size());
  }
  if (isotropic_only)
    UploadIsotropicFlux(phi);

  const std::uint64_t num_work_items = static_cast<std::uint64_t>(num_cells_) * ell_values.size() *
                                       static_cast<std::uint64_t>(groupset.GetNumGroups());
  if (num_work_items != 0)
  {
    constexpr std::uint32_t block_size = 256;
    crb::Dim3 block(block_size);
    crb::Dim3 grid = LaunchGrid(num_work_items, block_size, "GPU material-source");

    const MaterialSourceArguments args{nodes_.get(),
                                       cell_node_offsets_.get(),
                                       scatter_row_offsets_.get(),
                                       scatter_columns_.get(),
                                       scatter_values_.get(),
                                       production_.get(),
                                       fissionable_.get(),
                                       ell_values_.get(),
                                       ell_moment_offsets_.get(),
                                       ell_moments_.get(),
                                       isotropic_only ? phi_isotropic_.get() : phi_.get(),
                                       q_device,
                                       num_work_items,
                                       static_cast<std::uint32_t>(ell_values.size()),
                                       static_cast<std::uint32_t>(num_groups_),
                                       static_cast<std::uint32_t>(num_moments_),
                                       groupset.first_group,
                                       groupset.GetNumGroups(),
                                       isotropic_only,
                                       apply_wgs_scatter,
                                       apply_ags_scatter,
                                       apply_wgs_fission,
                                       apply_ags_fission,
                                       suppress_wg_scatter};
#if defined(__NVCC__) || defined(__HIPCC__)
    MaterialSourceKernel<<<grid, block>>>(args);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
    auto& stream = crb::Stream::get_null_stream();
    stream.parallel_for(sycl::nd_range<3>(grid * block, block),
                        [=](sycl::nd_item<3> work_index)
                        {
                          static_cast<void>(work_index);
                          MaterialSourceKernel(args);
                        });
#endif
  }
}

double
MaterialSourceCarrier::ComputeFissionProduction(const std::vector<double>& phi)
{
  if (phi.size() != vector_size_)
    throw std::invalid_argument(
      "GPU fission-production vector size does not match the LBS problem.");
  if (num_nodes_ == 0)
    return 0.0;

  UploadIsotropicFlux(phi);

  constexpr std::uint32_t block_size = 256;
  crb::Dim3 block(block_size);
  crb::Dim3 grid = LaunchGrid(num_nodes_, block_size, "GPU fission-production");
  const FissionProductionArguments args{nodes_.get(),
                                        production_column_sums_.get(),
                                        fissionable_.get(),
                                        phi_isotropic_.get(),
                                        node_production_.get(),
                                        num_nodes_,
                                        static_cast<std::uint32_t>(num_groups_)};
#if defined(__NVCC__) || defined(__HIPCC__)
  FissionProductionKernel<<<grid, block>>>(args);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto& stream = crb::Stream::get_null_stream();
  stream.parallel_for(sycl::nd_range<3>(grid * block, block),
                      [=](sycl::nd_item<3> work_index)
                      {
                        static_cast<void>(work_index);
                        FissionProductionKernel(args);
                      });
#endif

  {
    CaliperRegionScope transfer_scope("GPUTransfer", CaliperGPUTransferScopeDepth());
    crb::copy(node_production_host_, node_production_, num_nodes_);
  }
  return std::accumulate(node_production_host_.begin(), node_production_host_.end(), 0.0);
}

} // namespace opensn
