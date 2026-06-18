// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/source_xs_carrier.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"

namespace opensn
{

SourceXSCarrier::SourceXSCarrier(LBSProblem& lbs_problem)
{
  const std::uint64_t size = ComputeSize(lbs_problem);
  host_memory_.reserve(size);
  host_memory_.resize(size);
  device_memory_ = crb::DeviceMemory<char>(size);
  Assemble(lbs_problem);
  crb::copy(device_memory_, host_memory_, size);
}

std::uint64_t
SourceXSCarrier::ComputeSize(LBSProblem& lbs_problem)
{
  const auto& xs_map = lbs_problem.GetBlockID2XSMap();
  num_moments_ = lbs_problem.GetNumMoments();
  std::uint64_t size = xs_map.size() * sizeof(BlockData);

  for (const auto& [block_id, xs] : xs_map)
  {
    if (num_groups_ == std::numeric_limits<std::size_t>::max())
      num_groups_ = xs->GetNumGroups();
    else if (num_groups_ != xs->GetNumGroups())
      throw std::runtime_error("SourceXSCarrier: inconsistent group counts.");

    size += num_moments_ * num_groups_ * num_groups_ * sizeof(double);
    size += num_groups_ * num_groups_ * sizeof(double);
  }

  return size;
}

void
SourceXSCarrier::Assemble(LBSProblem& lbs_problem)
{
  const auto& xs_map = lbs_problem.GetBlockID2XSMap();
  char* host_base = host_memory_.data();
  char* device_base = device_memory_.get();
  auto* block_data = reinterpret_cast<BlockData*>(host_base);
  char* data = reinterpret_cast<char*>(block_data + xs_map.size());

  std::uint32_t block_index = 0;
  for (const auto& [block_id, xs] : xs_map)
  {
    block_id_to_index_[block_id] = block_index;
    auto& block = block_data[block_index++];
    block.num_moments = static_cast<std::uint32_t>(num_moments_);
    block.num_groups = static_cast<std::uint32_t>(num_groups_);
    block.has_scatter = not xs->GetTransferMatrices().empty();
    block.is_fissionable = xs->IsFissionable();

    auto* scatter = reinterpret_cast<double*>(data);
    block.scatter = reinterpret_cast<double*>(device_base + (data - host_base));
    std::fill(scatter, scatter + num_moments_ * num_groups_ * num_groups_, 0.0);
    const auto& transfer = xs->GetTransferMatrices();
    for (std::size_t m = 0; m < std::min(num_moments_, transfer.size()); ++m)
    {
      for (std::size_t g = 0; g < num_groups_; ++g)
        for (const auto& [_, gp, value] : transfer[m].Row(g))
          scatter[(m * num_groups_ + g) * num_groups_ + gp] = value;
    }
    data = reinterpret_cast<char*>(scatter + num_moments_ * num_groups_ * num_groups_);

    auto* production = reinterpret_cast<double*>(data);
    block.production = reinterpret_cast<double*>(device_base + (data - host_base));
    std::fill(production, production + num_groups_ * num_groups_, 0.0);
    if (block.is_fissionable)
    {
      const auto& F = xs->GetProductionMatrix();
      for (std::size_t g = 0; g < std::min(num_groups_, F.size()); ++g)
        for (std::size_t gp = 0; gp < std::min(num_groups_, F[g].size()); ++gp)
          production[g * num_groups_ + gp] = F[g][gp];
    }
    data = reinterpret_cast<char*>(production + num_groups_ * num_groups_);
  }
}

SourceXSCarrier::BlockData*
SourceXSCarrier::GetXSGPUData(int block_id)
{
  auto* data = reinterpret_cast<BlockData*>(device_memory_.get());
  return data + block_id_to_index_.at(block_id);
}

const SourceXSCarrier::BlockData*
SourceXSCarrier::GetXSHostData(int block_id) const
{
  const auto* data = reinterpret_cast<const BlockData*>(host_memory_.data());
  return data + block_id_to_index_.at(block_id);
}

} // namespace opensn
