// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/device/view/inline_macro.h"
#include <array>
#include <cstdint>

namespace opensn
{

/// Direction view.
struct DirectionView
{
  __inline_host_dev__ DirectionView() {}
  __inline_host_dev__ void Update(const double* direction_data, const std::uint32_t& num_moments)
  {
    omega[0] = direction_data[0];
    omega[1] = direction_data[1];
    omega[2] = direction_data[2];
    weight = direction_data[3];
    m2d = direction_data + 4;
    d2m = m2d + num_moments;
  }

  std::array<double, 3> omega;
  double weight;
  const double* m2d;
  const double* d2m;
};

/// Quadrature view.
struct QuadratureView
{
  __inline_host_dev__ QuadratureView(const char* quad_data)
  {
    // number of angles and number of moments
    const std::uint32_t* num_angles_and_moments_data =
      reinterpret_cast<const std::uint32_t*>(quad_data);
    num_angles = *(num_angles_and_moments_data++);
    num_moments = *(num_angles_and_moments_data++);
    quad_data = reinterpret_cast<const char*>(num_angles_and_moments_data);
    const std::uint32_t* moment_ell_ptr = reinterpret_cast<const std::uint32_t*>(quad_data);
    moment_ell = moment_ell_ptr;
    const std::uint32_t padded_num_moments = (num_moments + 1) & ~static_cast<std::uint32_t>(1);
    quad_data = reinterpret_cast<const char*>(moment_ell_ptr + padded_num_moments);
    // direction data
    direction_data = reinterpret_cast<const double*>(quad_data);
  }

  __inline_host_dev__ void GetDirectionView(DirectionView& direction, const std::uint32_t& index)
  {
    direction.Update(direction_data + index * (4 + 2 * num_moments), num_moments);
  }

  std::uint32_t num_angles;
  std::uint32_t num_moments;
  const std::uint32_t* moment_ell;
  const double* direction_data;
};

} // namespace opensn
