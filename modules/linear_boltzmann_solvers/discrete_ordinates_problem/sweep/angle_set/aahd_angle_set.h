// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/aahd_async_comm.h"
#include "caribou/main.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <set>

namespace crb = caribou;

namespace opensn
{

class AAHDSweepChunk;

/// AAHD angle set.
class AAHD_AngleSet : public AngleSet
{
public:
  AAHD_AngleSet(size_t id,
                const LBSGroupset& groupset,
                const SPDS& spds,
                std::shared_ptr<FLUDS>& fluds,
                std::vector<size_t>& angle_indices,
                std::map<uint64_t, std::shared_ptr<SweepBoundary>>& boundaries,
                int maximum_message_size,
                const MPICommunicatorSet& in_comm_set);

  crb::Stream& GetStream() { return stream_; }

  std::uint32_t* GetDeviceAngleIndices() { return device_angle_indices_.get(); }

  void InitializeDelayedUpstreamData() override;

  int GetMaxBufferMessages() const override { return async_comm_.GetMaxNumMessages(); }

  void SetMaxBufferMessages(int count) override { async_comm_.SetMaxNumMessages(count); }

  void PrepostReceives(bool use_device_buffers = false, bool delayed_psi_on_device = false);

  bool IsReady();

  void WaitForDownstreamAndDelayed();

  AngleSetStatus AngleSetAdvance(SweepChunk& sweep_chunk, AngleSetStatus permission) override;

  /// Upload non-local incoming psi and launch the sweep kernel without synchronizing.
  void LaunchSweepKernel(SweepChunk& sweep_chunk, bool incoming_psi_on_device = false);
  void LaunchSweepKernel(SweepChunk& sweep_chunk,
                         bool incoming_psi_on_device,
                         std::atomic<long long>* incoming_copy_time_ns);
  /// Synchronize the angle set's stream after a launched sweep kernel.
  void SynchronizeSweep(std::atomic<long long>* kernel_sync_time_ns = nullptr);

  /// Download non-local outgoing psi, signal intra-rank followers, and send downstream psi.
  void SendAfterFirstPass(bool use_device_buffers = false);
  /// Worker-side portion of the first-pass completion path.
  void PrepareAfterFirstPass(bool use_device_buffers,
                             std::atomic<long long>* copy_time_ns,
                             std::atomic<long long>* dependency_time_ns);
  /// Host-side MPI send issuance for data prepared by the first pass.
  void IssueDownstreamSends(bool use_device_buffers,
                            std::atomic<long long>* mpi_send_time_ns,
                            std::atomic<long long>* message_count,
                            std::atomic<long long>* total_doubles,
                            std::atomic<long long>* max_message_doubles);

  /// Final pass: download local delayed psi + save angular flux, copy to destination psi.
  void FinalizeAfterSweep(SweepChunk& sweep_chunk,
                          bool use_device_buffers = false,
                          bool final_download = true,
                          bool download_delayed_psi = true);

  AngleSetStatus FlushSendBuffers() override { return AngleSetStatus::MESSAGES_SENT; }

  void ResetSweepBuffers() override { executed_ = false; }

  bool ReceiveDelayedData() override { return true; }

  void SyncDeviceAngleIndices() override;

  void AddBoundaryOffset(std::uint64_t offset) { host_boundary_offsets_.push_back(offset); }

  void CopyBoundaryOffsetToDevice();

  std::uint64_t* GetDeviceBoudnaryOffset() { return device_boundary_offsets_.get(); }

  static std::mutex m;

protected:
  /// Associated CUDA stream.
  crb::Stream stream_ = crb::Stream::get_null_stream();

  /// Asynchronous communicator.
  AAHD_ASynchronousCommunicator async_comm_;

  /// Angle indices on GPU.
  crb::DeviceMemory<std::uint32_t> device_angle_indices_;

  /// Vector of offsets into the boundary bank.
  std::vector<std::uint64_t> host_boundary_offsets_;

  /// Boundary offset on GPU.
  crb::DeviceMemory<std::uint64_t> device_boundary_offsets_;

};

} // namespace opensn
