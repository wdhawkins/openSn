// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/sweep/boundary/reflecting_boundary.h"
#include "framework/logging/log.h"
#include "caliper/cali.h"

namespace opensn
{
namespace lbs
{

double*
ReflectingBoundary::PsiIncoming(uint64_t cell_local_id,
                                unsigned int face_num,
                                unsigned int fi,
                                unsigned int angle_num,
                                int group_num)
{
  int reflected_angle_num = reflected_anglenum_[angle_num];

  if (opposing_reflected_)
    return boundary_flux_old_[reflected_angle_num][cell_local_id][face_num][fi].data();

  return boundary_flux_[reflected_angle_num][cell_local_id][face_num][fi].data();
}

double*
ReflectingBoundary::PsiOutgoing(uint64_t cell_local_id,
                                unsigned int face_num,
                                unsigned int fi,
                                unsigned int angle_num)
{
  return boundary_flux_[angle_num][cell_local_id][face_num][fi].data();
}

void
ReflectingBoundary::UpdateAnglesReadyStatus(const std::vector<size_t>& angles)
{
  for (auto n : angles)
    angle_readyflags_[reflected_anglenum_[n]] = true;
}

bool
ReflectingBoundary::CheckAnglesReadyStatus(const std::vector<size_t>& angles)
{
  if (opposing_reflected_)
    return true;

  for (auto& n : angles)
    if (not boundary_flux_[reflected_anglenum_[n]].empty())
      if (not angle_readyflags_[n])
        return false;

  return true;
}

void
ReflectingBoundary::ResetAnglesReadyStatus()
{
  boundary_flux_old_ = boundary_flux_;
  angle_readyflags_.assign(angle_readyflags_.size(), false);
}

} // namespace lbs
} // namespace opensn
