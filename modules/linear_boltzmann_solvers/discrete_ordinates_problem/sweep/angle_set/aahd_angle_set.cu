// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include <algorithm>

namespace opensn
{

std::mutex AAHD_AngleSet::m;

AAHD_AngleSet::AAHD_AngleSet(size_t id,
                             const LBSGroupset& groupset,
                             const SPDS& spds,
                             std::shared_ptr<FLUDS>& fluds,
                             std::vector<size_t>& angle_indices,
                             std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                             int maximum_message_size,
                             const MPICommunicatorSet& comm_set)
  : AngleSet(id, groupset, spds, fluds, angle_indices, boundaries),
    async_comm_(*fluds, num_groups_, angle_indices.size(), maximum_message_size, comm_set),
    device_angle_indices_(angles_.size())
{
  stream_ = crb::Stream();
  auto aahd_fluds = std::dynamic_pointer_cast<AAHD_FLUDS>(fluds_);
  aahd_fluds->GetStream() = stream_;
  aahd_fluds->GetCommonData().AddAssociatedAngleSet(this);
  SyncDeviceAngleIndices();
}

void
AAHD_AngleSet::InitializeDelayedUpstreamData()
{
  auto* aahd_fluds = static_cast<AAHD_FLUDS*>(fluds_.get());
  aahd_fluds->AllocateDelayedPrelocIOutgoingPsi();
  aahd_fluds->AllocateDelayedLocalPsi();
}

void
AAHD_AngleSet::PrepostReceives(bool use_device_buffers, bool delayed_psi_on_device)
{
  async_comm_.PrepostReceiveUpstreamPsi(static_cast<int>(this->GetID()), use_device_buffers);
  async_comm_.PrepostReceiveDelayedData(static_cast<int>(this->GetID()), use_device_buffers);
  auto* aahd_fluds = static_cast<AAHD_FLUDS*>(fluds_.get());
  if (not delayed_psi_on_device)
    aahd_fluds->CopyDelayedPsiToDevice();
}

void
AAHD_AngleSet::WaitForDownstreamAndDelayed()
{
  async_comm_.WaitForDownstreamPsi();
  async_comm_.WaitForDelayedIncomingPsi();
}

bool
AAHD_AngleSet::IsReady()
{
  return (async_comm_.TestReceiveUpstreamPsi()) && (dependency_counter_ == 0);
}

AngleSetStatus
AAHD_AngleSet::AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission)
{
  if (executed_)
    return AngleSetStatus::FINISHED;
  SweepKernelAndSync(sweep_chunk);
  SendAfterFirstPass();
  FinalizeAfterSweep(sweep_chunk);
  return AngleSetStatus::FINISHED;
}

void
AAHD_AngleSet::SweepKernelAndSync(SweepChunk& sweep_chunk, bool incoming_psi_on_device)
{
  auto* aahd_fluds = static_cast<AAHD_FLUDS*>(fluds_.get());
  auto& aahd_sweep_chunk = static_cast<AAHDSweepChunk&>(sweep_chunk);
  if (not incoming_psi_on_device)
    aahd_fluds->CopyNonLocalIncomingPsiToDevice();
  aahd_fluds->AllocateSaveAngularFlux(aahd_sweep_chunk.GetProblem(),
                                      aahd_sweep_chunk.GetGroupset());
  aahd_sweep_chunk.Sweep(*this);
  stream_.synchronize();
}

void
AAHD_AngleSet::SendAfterFirstPass(bool use_device_buffers)
{
  SendAfterFirstPass(use_device_buffers, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void
AAHD_AngleSet::SendAfterFirstPass(bool use_device_buffers,
                                  std::atomic<long long>* copy_time_ns,
                                  std::atomic<long long>* dependency_time_ns,
                                  std::atomic<long long>* mpi_send_time_ns,
                                  std::atomic<long long>* message_count,
                                  std::atomic<long long>* total_doubles,
                                  std::atomic<long long>* max_message_doubles)
{
  using Clock = std::chrono::high_resolution_clock;

  auto* aahd_fluds = static_cast<AAHD_FLUDS*>(fluds_.get());
  if (aahd_fluds->HasNonLocalOutgoingPsi() and not use_device_buffers)
  {
    const auto copy_start = Clock::now();
    aahd_fluds->CopyNonLocalOutgoingPsiFromDevice();
    stream_.synchronize();
    if (copy_time_ns != nullptr)
      copy_time_ns->fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - copy_start).count(),
        std::memory_order_relaxed);
  }
  if (not following_angle_sets_.empty())
  {
    const auto dependency_start = Clock::now();
    std::scoped_lock lk(m);
    for (auto& following_as : following_angle_sets_)
      following_as->DecrementCounter();
    if (dependency_time_ns != nullptr)
      dependency_time_ns->fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      Clock::now() - dependency_start)
                                      .count(),
                                    std::memory_order_relaxed);
  }
  const auto mpi_send_start = Clock::now();
  if (use_device_buffers)
  {
    std::scoped_lock lk(m);
    async_comm_.SendDownstreamPsi(static_cast<int>(this->GetID()),
                                  use_device_buffers,
                                  message_count,
                                  total_doubles,
                                  max_message_doubles);
  }
  else
    async_comm_.SendDownstreamPsi(static_cast<int>(this->GetID()),
                                  use_device_buffers,
                                  message_count,
                                  total_doubles,
                                  max_message_doubles);
  if (mpi_send_time_ns != nullptr)
    mpi_send_time_ns->fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - mpi_send_start).count(),
      std::memory_order_relaxed);
}

void
AAHD_AngleSet::FinalizeAfterSweep(SweepChunk& sweep_chunk,
                                  bool use_device_buffers,
                                  bool final_download,
                                  bool download_delayed_psi)
{
  auto* aahd_fluds = static_cast<AAHD_FLUDS*>(fluds_.get());
  auto& aahd_sweep_chunk = static_cast<AAHDSweepChunk&>(sweep_chunk);
  // Non-local outgoing psi was already downloaded and sent in SendAfterFirstPass().
  // Here we download local delayed psi and saved angular flux.
  const bool need_local_delayed_download = aahd_fluds->HasAnyLocalDelayedPsi();
  if (download_delayed_psi and need_local_delayed_download)
    aahd_fluds->CopyLocalDelayedPsiFromDevice();
  if (final_download)
    aahd_fluds->CopySaveAngularFluxFromDevice();
  if ((download_delayed_psi and need_local_delayed_download) or
      (final_download and aahd_fluds->HasSaveAngularFlux()))
    stream_.synchronize();
  if (final_download)
    aahd_fluds->CopySaveAngularFluxToDestinationPsi(
      aahd_sweep_chunk.GetProblem(), aahd_sweep_chunk.GetGroupset(), *this);
  executed_ = true;
}

void
AAHD_AngleSet::SyncDeviceAngleIndices()
{
  crb::copy(device_angle_indices_, angles_, angles_.size());
}

void
AAHD_AngleSet::CopyBoundaryOffsetToDevice()
{
  if (host_boundary_offsets_.empty())
  {
    device_boundary_offsets_.reset();
    return;
  }
  device_boundary_offsets_ = crb::DeviceMemory<std::uint64_t>(host_boundary_offsets_.size());
  crb::copy(device_boundary_offsets_, host_boundary_offsets_, host_boundary_offsets_.size());
  host_boundary_offsets_.clear();
}

} // namespace opensn
