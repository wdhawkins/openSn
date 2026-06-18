// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once
#include "caribou/main.hpp"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/boundary/sweep_boundary.h"
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace crb = caribou;

namespace opensn
{

class BoundaryBank;
class LBSGroupset;

/**
 * Object managing boundary flux data on GPU.
 * This class pins the host static boundary flux banks owned by SweepBoundary and allocates matching
 * device memory for each groupset.
 */
class BoundaryCarrier
{
public:
  /// Constructor from the list of groupsets.
  BoundaryCarrier(BoundaryBank& bank,
                  const std::vector<LBSGroupset>& groupsets,
                  const std::map<std::uint64_t, std::shared_ptr<SweepBoundary>>& boundaries);

  /// Copy the boundary flux data for the specified groupset from host to device.
  void UploadToDevice(int groupset_id);
  /// Copy the boundary flux data for the specified groupset from device to host.
  void DownloadToHost(int groupset_id);
  /// Copy delayed reflecting boundary current fluxes to their old-flux ranges on device.
  void CopyDelayedAngularFluxNewToOldOnDevice(int groupset_id);
  /// Compute device-side pointwise relative change for delayed reflecting boundary fluxes.
  double ComputeDelayedAngularFluxPointwiseChangeOnDevice(int groupset_id);
  /// Get device pointer of a given groupset.
  double* GetDevicePtr(int groupset_id) { return device_boundary_flux_[groupset_id].get(); }

protected:
  /// Boundary flux storage indexed by groupset ID.
  std::vector<crb::DeviceMemory<double>> device_boundary_flux_;
  /// Delayed boundary flux ranges indexed by groupset ID.
  std::vector<std::vector<BoundaryDelayedAngularFluxRange>> delayed_ranges_;
  /// Reference to boundary bank.
  BoundaryBank& bank_;
};

} // namespace opensn
