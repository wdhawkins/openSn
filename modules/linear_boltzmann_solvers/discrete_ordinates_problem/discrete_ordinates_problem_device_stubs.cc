// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/utils/error.h"

namespace opensn
{

#ifndef __OPENSN_WITH_GPU__

void
DiscreteOrdinatesProblem::CopyDeviceBoundaryDelayedPsiNewToOld(int)
{
  OpenSnLogicalError("CopyDeviceBoundaryDelayedPsiNewToOld requires GPU support.");
}

double
DiscreteOrdinatesProblem::ComputeDeviceBoundaryDelayedPsiPointwiseChange(int)
{
  OpenSnLogicalError("ComputeDeviceBoundaryDelayedPsiPointwiseChange requires GPU support.");
}

#endif

} // namespace opensn
