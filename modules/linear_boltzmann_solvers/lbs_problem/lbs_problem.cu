// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/mesh_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/material_source_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/total_xs_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/device/device_vector_mirror.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/outflow/outflow_carrier.h"
#include "framework/utils/error.h"
#include "framework/utils/caliper_scopes.h"

#include "caribou/main.hpp"
namespace crb = caribou;

namespace opensn
{

namespace
{

bool
CheckIntegratedMemory()
{
#ifdef __NVCC__
  int device_id;
  crb::impl::check_error(::cudaGetDevice(&device_id));
  int val = 0;
  crb::impl::check_error(::cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, device_id));
  return val != 0;
#elif defined(__HIPCC__)
  int device_id;
  crb::impl::check_error(::hipGetDevice(&device_id));
  int val = 0;
  crb::impl::check_error(::hipDeviceGetAttribute(&val, hipDeviceAttributeIntegrated, device_id));
  return val != 0;
#else
  return false;
#endif
}

} // namespace

const bool LBSProblem::has_integrated_memory = CheckIntegratedMemory();

void
LBSProblem::InitializeGPUExtras()
{
  // exit if GPU acceleration is not enabled
  if (!use_gpus_)
  {
    return;
  }
  CaliperRegionScope cali_gpu_setup_scope("GPUSetup", CaliperGPUSetupScopeDepth());
  // initialize carriers
  total_xs_carrier_ = std::make_shared<TotalXSCarrier>(*this);
  outflow_carrier_ = std::make_shared<OutflowCarrier>(*this);
  mesh_carrier_ = std::make_shared<MeshCarrier>(*this, *total_xs_carrier_, *outflow_carrier_);
  material_source_carrier_ = std::make_shared<MaterialSourceCarrier>(*this);
  // initialize pinners
  source_pinner_ = std::make_shared<DeviceVectorMirror<double>>(q_moments_local_);
  phi_pinner_ = std::make_shared<DeviceVectorMirror<double>>(phi_new_local_);
}

void
LBSProblem::ResetGPUCarriers()
{
  // exit if GPU acceleration is not enabled
  if (!use_gpus_)
  {
    return;
  }
  // delete carriers
  total_xs_carrier_.reset();
  outflow_carrier_.reset();
  mesh_carrier_.reset();
  material_source_carrier_.reset();
  // delete pinners
  source_pinner_.reset();
  phi_pinner_.reset();
}

void
LBSProblem::RebuildOutflowDependentGPUCarriers()
{
  // exit if GPU acceleration is not enabled
  if (not use_gpus_)
    return;

  if (not total_xs_carrier_)
  {
    ResetGPUCarriers();
    InitializeGPUExtras();
    return;
  }
  outflow_carrier_.reset();
  mesh_carrier_.reset();
  outflow_carrier_ = std::make_shared<OutflowCarrier>(*this);
  mesh_carrier_ = std::make_shared<MeshCarrier>(*this, *total_xs_carrier_, *outflow_carrier_);
}

bool
LBSProblem::HasCapableDevices()
{
  return crb::get_num_gpus() > 0;
}

bool
LBSProblem::AddMaterialSourceGPU(const LBSGroupset& groupset,
                                 std::vector<double>& q,
                                 const std::vector<double>& phi,
                                 const SourceFlags source_flags,
                                 const bool copy_to_host) const
{
  const bool fixed_source_moments =
    (source_flags & APPLY_FIXED_SOURCES) and options_.use_src_moments;
  const bool delayed_fission =
    options_.use_precursors and
    ((source_flags & APPLY_WGS_FISSION_SOURCES) or (source_flags & APPLY_AGS_FISSION_SOURCES));
  if (not use_gpus_ or not material_source_carrier_ or IsTimeDependent() or fixed_source_moments or
      delayed_fission or &q != &q_moments_local_)
    return false;

  {
    CaliperRegionScope transfer_scope("GPUTransfer", CaliperGPUTransferScopeDepth());
    // A zeroed destination needs no upload; the host copy is overwritten below or left stale.
    if (source_flags & ZERO_SOURCE_MOMENTS)
      source_pinner_->ZeroOnDevice();
    else
      source_pinner_->CopyToDevice();
  }
  material_source_carrier_->Apply(groupset, q, phi, source_pinner_->GetDevicePtr(), source_flags);
  if (copy_to_host)
  {
    CaliperRegionScope transfer_scope("GPUTransfer", CaliperGPUTransferScopeDepth());
    source_pinner_->CopyFromDevice();
  }
  return true;
}

void
LBSProblem::CopySourceMomentsToDevice(const std::vector<double>& q) const
{
  if (not use_gpus_)
    return;
  OpenSnLogicalErrorIf(&q != &q_moments_local_,
                       "Only the problem's source moments have a device copy.");
  CaliperRegionScope transfer_scope("GPUTransfer", CaliperGPUTransferScopeDepth());
  source_pinner_->CopyToDevice();
}

bool
LBSProblem::ComputeFissionProductionGPU(const std::vector<double>& phi,
                                        double& local_production) const
{
  if (not use_gpus_ or not material_source_carrier_ or options_.use_precursors)
    return false;

  local_production = material_source_carrier_->ComputeFissionProduction(phi);
  return true;
}

} // namespace opensn
