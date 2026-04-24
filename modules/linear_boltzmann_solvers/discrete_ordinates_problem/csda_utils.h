// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <utility>
#include <vector>

namespace opensn
{

inline constexpr double kCSDATolerance = 1.0e-12;

inline std::vector<std::pair<unsigned int, unsigned int>>
FindCSDAChargedGroupRanges(const std::vector<double>& stopping_power)
{
  std::vector<std::pair<unsigned int, unsigned int>> ranges;

  unsigned int g = 0;
  while (g < stopping_power.size())
  {
    while (g < stopping_power.size() and std::abs(stopping_power[g]) <= kCSDATolerance)
      ++g;
    if (g >= stopping_power.size())
      break;

    const unsigned int g_begin = g;
    while (g < stopping_power.size() and std::abs(stopping_power[g]) > kCSDATolerance)
      ++g;
    ranges.emplace_back(g_begin, g);
  }

  return ranges;
}

inline double
CSDAGroupCenterEnergy(const std::vector<double>& energy_bounds, const unsigned int g)
{
  return 0.5 * (energy_bounds.at(g) + energy_bounds.at(g + 1));
}

} // namespace opensn
