// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/aah.h"
#include "caribou/main.hpp"

namespace crb = caribou;

namespace opensn
{

void
AAH_SPDS::CopySPLSDataOnDevice()
{
  // compute offset and total size to allocate on GPU
  contiguous_offset_.reserve(levelized_spls_.size());
  std::size_t total_size = 0;
  for (const std::vector<uint32_t>& level : levelized_spls_)
  {
    contiguous_offset_.push_back(total_size);
    total_size += level.size();
  }
  // allocate and copy data
  crb::DeviceMemory<std::uint32_t> device_levelized_spls(total_size);
  for (std::size_t l = 0; l < levelized_spls_.size(); ++l)
  {
    const std::vector<std::uint32_t>& level = levelized_spls_[l];
    crb::HostVector<std::uint32_t> host_level(level.begin(), level.end());
    crb::copy(device_levelized_spls, host_level, level.size(), 0, contiguous_offset_[l]);
  }
  // release ownership back to the class
  device_levelized_spls_ = device_levelized_spls.release();

  // Build per-level offset and size arrays for the persistent sweep kernel.
  // device_level_offsets_[l] == contiguous_offset_[l] and also == level_abs_starts_[l]
  // from AAHD_FLUDSCommonData, so one device array serves both indexing roles.
  const std::size_t num_levels = levelized_spls_.size();
  crb::HostVector<std::uint32_t> level_offsets_h(num_levels);
  crb::HostVector<std::uint32_t> level_sizes_h(num_levels);
  max_level_size_ = 0;
  for (std::size_t l = 0; l < num_levels; ++l)
  {
    level_offsets_h[l] = static_cast<std::uint32_t>(contiguous_offset_[l]);
    level_sizes_h[l] = static_cast<std::uint32_t>(levelized_spls_[l].size());
    if (level_sizes_h[l] > max_level_size_)
      max_level_size_ = level_sizes_h[l];
  }
  crb::DeviceMemory<std::uint32_t> lo(num_levels);
  crb::copy(lo, level_offsets_h, num_levels);
  device_level_offsets_ = lo.release();
  crb::DeviceMemory<std::uint32_t> ls(num_levels);
  crb::copy(ls, level_sizes_h, num_levels);
  device_level_sizes_ = ls.release();
}

void
AAH_SPDS::FreeDeviceData()
{
  if (device_levelized_spls_)
  {
    crb::DeviceMemory<std::uint32_t> device_levelized_spls(device_levelized_spls_);
    device_levelized_spls.reset();
    device_levelized_spls_ = nullptr;
    contiguous_offset_.clear();
  }
  if (device_level_offsets_)
  {
    crb::DeviceMemory<std::uint32_t> p(device_level_offsets_);
    p.reset();
    device_level_offsets_ = nullptr;
  }
  if (device_level_sizes_)
  {
    crb::DeviceMemory<std::uint32_t> p(device_level_sizes_);
    p.reset();
    device_level_sizes_ = nullptr;
  }
  max_level_size_ = 0;
}

} // namespace opensn
