// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/aahd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/spds.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aahd_fluds.h"
#include <algorithm>
#include <functional>
#include <numeric>

namespace opensn
{

static inline void
ResizeRequestVector(std::vector<mpi::Request>& request_vector,
                    const std::vector<std::vector<AAH_MessageDetails>>& message_data)
{
  std::size_t total_messages = std::transform_reduce(message_data.begin(),
                                                     message_data.end(),
                                                     0UL,
                                                     std::plus<>{},
                                                     [](const auto& v) { return v.size(); });
  request_vector.resize(total_messages);
}

AAHD_ASynchronousCommunicator::AAHD_ASynchronousCommunicator(FLUDS& fluds,
                                                             unsigned int num_groups,
                                                             std::size_t num_angles,
                                                             int max_mpi_message_size,
                                                             const MPICommunicatorSet& comm_set)
  : AsynchronousCommunicator(fluds, comm_set),
    max_num_messages_(0),
    max_mpi_message_size_(max_mpi_message_size)
{
  BuildMessageStructure();
}

void
AAHD_ASynchronousCommunicator::BuildMessageStructure()
{
  const auto& spds = fluds_.GetSPDS();
  auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&fluds_);
  if (aahd_fluds == nullptr)
    throw std::runtime_error("AAHD_ASynchronousCommunicator does not get AAHD_FLUDS.\n");
  // Predecessor locations
  SetupMessageData(
    spds.GetLocationDependencies(),
    [aahd_fluds](std::size_t i) { return aahd_fluds->GetNonLocalIncomingNumUnknowns(i); },
    preloc_msg_data_,
    nullptr,
    false,
    max_num_messages_,
    comm_set_,
    max_mpi_message_size_);
  ResizeRequestVector(preloc_msg_request_, preloc_msg_data_);
  // Delayed predecessor locations
  SetupMessageData(
    spds.GetDelayedLocationDependencies(),
    [aahd_fluds](std::size_t i) { return aahd_fluds->GetNonLocalDelayedIncomingNumUnknowns(i); },
    delayed_preloc_msg_data_,
    nullptr,
    false,
    max_num_messages_,
    comm_set_,
    max_mpi_message_size_);
  ResizeRequestVector(delayed_preloc_msg_request_, delayed_preloc_msg_data_);
  // Successor locations
  SetupMessageData(
    spds.GetLocationSuccessors(),
    [aahd_fluds](std::size_t i) { return aahd_fluds->GetNonLocalOutgoingNumUnknowns(i); },
    deploc_msg_data_,
    nullptr,
    true,
    max_num_messages_,
    comm_set_,
    max_mpi_message_size_);
  ResizeRequestVector(deploc_msg_request_, deploc_msg_data_);

  const int regular_max_num_messages = max_num_messages_;
  int promoted_max_num_messages = 0;
  const auto& common_data = aahd_fluds->GetCommonData();
  SetupMessageData(
    common_data.GetPromotedDelayedIncomingLocations(),
    [aahd_fluds](std::size_t i) { return aahd_fluds->GetPromotedDelayedIncomingNumUnknowns(i); },
    promoted_delayed_incoming_msg_data_,
    nullptr,
    false,
    promoted_max_num_messages,
    comm_set_,
    max_mpi_message_size_);
  ResizeRequestVector(promoted_delayed_incoming_msg_request_, promoted_delayed_incoming_msg_data_);
  SetupMessageData(
    common_data.GetPromotedDelayedOutgoingLocations(),
    [aahd_fluds](std::size_t i) { return aahd_fluds->GetPromotedDelayedOutgoingNumUnknowns(i); },
    promoted_delayed_outgoing_msg_data_,
    nullptr,
    true,
    promoted_max_num_messages,
    comm_set_,
    max_mpi_message_size_);
  ResizeRequestVector(promoted_delayed_outgoing_msg_request_, promoted_delayed_outgoing_msg_data_);

  const int tag_half_block = std::max(regular_max_num_messages, promoted_max_num_messages);
  promoted_msg_tag_offset_ = tag_half_block;
  max_num_messages_ = 2 * tag_half_block;
}

void
AAHD_ASynchronousCommunicator::PrepostReceiveUpstreamPsi(int angle_set_num,
                                                         bool use_device_buffers)
{

  const auto& spds = fluds_.GetSPDS();
  const auto& comm = comm_set_.LocICommunicator(opensn::mpi_comm.rank());
  const std::size_t num_dependencies = spds.GetLocationDependencies().size();
  auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&fluds_);

  for (std::size_t i = 0, req = 0; i < num_dependencies; ++i)
  {
    auto& upstream_psi = fluds_.PrelocIOutgoingPsi()[i];

    for (int m = 0; m < preloc_msg_data_[i].size(); ++m, ++req)
    {
      const auto& [source, size, block_pos] = preloc_msg_data_[i][m];
      int tag = max_num_messages_ * angle_set_num + m;
      double* buffer = &upstream_psi[block_pos];
      if (use_device_buffers)
      {
        aahd_fluds->ValidateDevicePrelocIOutgoingPsi(i, block_pos, size);
        buffer = aahd_fluds->DevicePrelocIOutgoingPsi(i, block_pos);
      }
      preloc_msg_request_[req] = comm.irecv(source, tag, buffer, size);
    }
  }
}

bool
AAHD_ASynchronousCommunicator::TestReceiveUpstreamPsi()
{
  return mpi::test_all(preloc_msg_request_);
}

void
AAHD_ASynchronousCommunicator::WaitForUpstreamPsi()
{
  mpi::wait_all(preloc_msg_request_);
}

void
AAHD_ASynchronousCommunicator::PrepostReceiveDelayedData(int angle_set_num,
                                                         bool use_device_buffers)
{

  const auto& spds = fluds_.GetSPDS();
  const auto& comm = comm_set_.LocICommunicator(opensn::mpi_comm.rank());
  const std::size_t num_delayed_dependencies = spds.GetDelayedLocationDependencies().size();
  auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&fluds_);

  for (std::size_t i = 0, req = 0; i < num_delayed_dependencies; ++i)
  {
    auto& upstream_psi = fluds_.DelayedPrelocIOutgoingPsi()[i];

    for (int m = 0; m < delayed_preloc_msg_data_[i].size(); ++m, ++req)
    {
      const auto& [source, size, block_pos] = delayed_preloc_msg_data_[i][m];
      int tag = max_num_messages_ * angle_set_num + m;
      double* buffer = &upstream_psi[block_pos];
      if (use_device_buffers)
      {
        aahd_fluds->ValidateDeviceDelayedPrelocIOutgoingPsi(i, block_pos, size);
        buffer = aahd_fluds->DeviceDelayedPrelocIOutgoingPsi(i, block_pos);
      }
      delayed_preloc_msg_request_[req] = comm.irecv(source, tag, buffer, size);
    }
  }

  auto& promoted_incoming_psi = aahd_fluds->PromotedDelayedIncomingPsi();
  for (std::size_t i = 0, req = 0; i < promoted_delayed_incoming_msg_data_.size(); ++i)
  {
    auto& upstream_psi = promoted_incoming_psi[i];
    for (int m = 0; m < promoted_delayed_incoming_msg_data_[i].size(); ++m, ++req)
    {
      const auto& [source, size, block_pos] = promoted_delayed_incoming_msg_data_[i][m];
      int tag = max_num_messages_ * angle_set_num + promoted_msg_tag_offset_ + m;
      double* buffer = &upstream_psi[block_pos];
      if (use_device_buffers)
      {
        aahd_fluds->ValidateDevicePromotedDelayedIncomingPsi(i, block_pos, size);
        buffer = aahd_fluds->DevicePromotedDelayedIncomingPsi(i, block_pos);
      }
      promoted_delayed_incoming_msg_request_[req] =
        comm.irecv(source, tag, buffer, size);
    }
  }
}

void
AAHD_ASynchronousCommunicator::WaitForDelayedIncomingPsi()
{
  mpi::wait_all(delayed_preloc_msg_request_);
  mpi::wait_all(promoted_delayed_incoming_msg_request_);
}

void
AAHD_ASynchronousCommunicator::SendDownstreamPsi(int angle_set_num, bool use_device_buffers)
{

  const auto& spds = fluds_.GetSPDS();
  const auto& location_successors = spds.GetLocationSuccessors();
  const std::size_t num_successors = location_successors.size();
  auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&fluds_);

  for (std::size_t i = 0, req = 0; i < num_successors; ++i)
  {
    const auto& comm = comm_set_.LocICommunicator(location_successors[i]);
    const auto& outgoing_psi = fluds_.DeplocIOutgoingPsi()[i];

    for (int m = 0; m < deploc_msg_data_[i].size(); ++m, ++req)
    {
      const auto& [dest, size, block_pos] = deploc_msg_data_[i][m];
      int tag = max_num_messages_ * angle_set_num + m;
      double* buffer = &outgoing_psi[block_pos];
      if (use_device_buffers)
      {
        aahd_fluds->ValidateDeviceDeplocIOutgoingPsi(i, block_pos, size);
        buffer = aahd_fluds->DeviceDeplocIOutgoingPsi(i, block_pos);
      }
      deploc_msg_request_[req] = comm.isend(dest, tag, buffer, size);
    }
  }
}

void
AAHD_ASynchronousCommunicator::SendPromotedDelayedPsi(int angle_set_num,
                                                      bool use_device_buffers)
{
  auto* aahd_fluds = dynamic_cast<AAHD_FLUDS*>(&fluds_);
  const auto& promoted_locations = aahd_fluds->GetCommonData().GetPromotedDelayedOutgoingLocations();
  auto& promoted_outgoing_psi = aahd_fluds->PromotedDelayedOutgoingPsi();

  for (std::size_t i = 0, req = 0; i < promoted_locations.size(); ++i)
  {
    const auto& comm = comm_set_.LocICommunicator(promoted_locations[i]);
    const auto& outgoing_psi = promoted_outgoing_psi[i];

    for (int m = 0; m < promoted_delayed_outgoing_msg_data_[i].size(); ++m, ++req)
    {
      const auto& [dest, size, block_pos] = promoted_delayed_outgoing_msg_data_[i][m];
      int tag = max_num_messages_ * angle_set_num + promoted_msg_tag_offset_ + m;
      double* buffer = &outgoing_psi[block_pos];
      if (use_device_buffers)
      {
        aahd_fluds->ValidateDevicePromotedDelayedOutgoingPsi(i, block_pos, size);
        buffer = aahd_fluds->DevicePromotedDelayedOutgoingPsi(i, block_pos);
      }
      promoted_delayed_outgoing_msg_request_[req] =
        comm.isend(dest, tag, buffer, size);
    }
  }
}

void
AAHD_ASynchronousCommunicator::WaitForPromotedDelayedPsi()
{
  mpi::wait_all(promoted_delayed_outgoing_msg_request_);
}

void
AAHD_ASynchronousCommunicator::WaitForDownstreamPsi()
{
  mpi::wait_all(deploc_msg_request_);
}

} // namespace opensn
