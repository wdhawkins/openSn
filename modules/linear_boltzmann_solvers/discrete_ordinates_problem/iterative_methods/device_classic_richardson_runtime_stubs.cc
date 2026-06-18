// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/device_classic_richardson_runtime.h"
#include "framework/utils/error.h"

namespace opensn
{

#ifndef __OPENSN_WITH_GPU__

class DiscreteOrdinatesProblem;
class LBSGroupset;
struct SweepWGSContext;

DeviceClassicRichardsonRuntime::DeviceClassicRichardsonRuntime(SweepWGSContext&)
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::CopyPhiOldToDevice()
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::CopySourceMomentsToDevice()
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::CopyDevicePhiNewToOld()
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::CopySourceBaseToDevice()
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::UploadDelayedPsiToDevice()
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::BuildSource(const LBSGroupset&, bool, bool)
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

double
DeviceClassicRichardsonRuntime::ComputePhiChange(const LBSGroupset&) const
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::ApplyInverseTransportOperator(SourceFlags)
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

void
DeviceClassicRichardsonRuntime::FinalizeAngularFluxes()
{
  OpenSnLogicalError("DeviceClassicRichardsonRuntime requires GPU support.");
}

#endif

} // namespace opensn
