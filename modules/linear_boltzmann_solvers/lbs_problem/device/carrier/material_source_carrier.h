// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/source_functions/source_flags.h"
#include "caribou/main.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace crb = caribou;

namespace opensn
{

class LBSGroupset;
class LBSProblem;

/** Device-resident cross sections and reusable work buffers for material-source assembly. */
class MaterialSourceCarrier
{
public:
  struct NodeData
  {
    std::uint64_t phi_address;
    std::uint32_t material_index;
    double volume_weight;
  };

  explicit MaterialSourceCarrier(const LBSProblem& lbs_problem);

  /// Adds the scattering and prompt fission sources of `source_flags` to `q_device`, the device
  /// copy of `q`. The launch is asynchronous on the null stream.
  void Apply(const LBSGroupset& groupset,
             const std::vector<double>& q,
             const std::vector<double>& phi,
             double* q_device,
             SourceFlags source_flags);

  /// Returns the rank-local prompt fission production of `phi`.
  double ComputeFissionProduction(const std::vector<double>& phi);

private:
  /// Uploads the isotropic moment of `phi`, node by node, to `phi_isotropic_`.
  void UploadIsotropicFlux(const std::vector<double>& phi);

  std::size_t num_groups_ = 0;
  std::size_t num_moments_ = 0;
  std::size_t num_nodes_ = 0;
  std::size_t num_cells_ = 0;
  std::size_t vector_size_ = 0;

  crb::DeviceMemory<NodeData> nodes_;
  /// Offset of each local cell's first node in `nodes_`, plus the total.
  crb::DeviceMemory<std::uint32_t> cell_node_offsets_;
  std::vector<std::uint64_t> node_phi_addresses_;
  crb::DeviceMemory<std::uint64_t> scatter_row_offsets_;
  crb::DeviceMemory<std::uint32_t> scatter_columns_;
  crb::DeviceMemory<double> scatter_values_;
  crb::DeviceMemory<double> production_;
  /// Per-material sums over destination groups of the production matrix, by source group.
  crb::DeviceMemory<double> production_column_sums_;
  crb::DeviceMemory<std::uint8_t> fissionable_;
  /// Moments grouped by harmonic degree for the current Apply call.
  crb::DeviceMemory<std::uint32_t> ell_values_;
  crb::DeviceMemory<std::uint32_t> ell_moment_offsets_;
  crb::DeviceMemory<std::uint32_t> ell_moments_;
  crb::DeviceMemory<double> phi_;
  std::vector<double> phi_isotropic_host_;
  crb::DeviceMemory<double> phi_isotropic_;
  crb::DeviceMemory<double> node_production_;
  std::vector<double> node_production_host_;
};

} // namespace opensn
