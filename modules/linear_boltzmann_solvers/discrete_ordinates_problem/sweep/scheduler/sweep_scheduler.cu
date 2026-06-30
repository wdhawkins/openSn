// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "caribou/main.hpp"
#include "framework/utils/error.h"
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>

namespace opensn
{

namespace
{

bool
UseFixedPointSweepDeviceMPI()
{
  const char* value = std::getenv("OPENSN_FIXED_POINT_SWEEP_DEVICE_MPI");
  return value == nullptr or (std::string(value) != "0" and std::string(value) != "false");
}

} // namespace

void
SweepScheduler::ScheduleAlgoAAO(SweepChunk& sweep_chunk)
{
  auto& aah_sweep_chunk = static_cast<AAHDSweepChunk&>(sweep_chunk);
  auto& groupset = aah_sweep_chunk.GetGroupset();
  int groupset_id = angle_agg_.GetGroupsetID();
  const bool force_final_angular_flux_pullback = force_final_angular_flux_pullback_;
  force_final_angular_flux_pullback_ = false;
  const bool defer_angular_flux_pullback =
    groupset.fixed_point_sweep and groupset.fixed_point_sweep_defer_angular_flux_pullback and
    not force_final_angular_flux_pullback;
  const bool pullback_angular_flux_on_final_pass = not defer_angular_flux_pullback;
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, true);

  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();

  auto execute_pass = [&](bool use_device_buffers, bool delayed_psi_on_device, bool final_download)
  {
    execution_order_.resize(num_anglesets);
    for (std::size_t i = 0; i < num_anglesets; ++i)
    {
      auto* aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      aahd_angle_set->ResetDependencyCounter();
      aahd_angle_set->PrepostReceives(use_device_buffers, delayed_psi_on_device);
      execution_order_[i] = i;
    }

    pool_.AssignTask(
      [this, &sweep_chunk, use_device_buffers, final_download](std::size_t i)
      {
        auto* aahd = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
        aahd->SweepKernelAndSync(sweep_chunk, use_device_buffers);
        aahd->SendAfterFirstPass(use_device_buffers);
        aahd->FinalizeAfterSweep(sweep_chunk, use_device_buffers, final_download);
      });

    bool all_angle_sets_ready = true;
    for (std::size_t i = 0; i < num_anglesets; ++i)
    {
      auto* angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      if (not angle_set->IsReady())
      {
        all_angle_sets_ready = false;
        break;
      }
    }
    if (all_angle_sets_ready)
    {
      pool_.ExecuteBatch(
        [this, &sweep_chunk, use_device_buffers, final_download](std::size_t i)
        {
          auto* aahd = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
          aahd->SweepKernelAndSync(sweep_chunk, use_device_buffers);
          aahd->SendAfterFirstPass(use_device_buffers);
          aahd->FinalizeAfterSweep(sweep_chunk, use_device_buffers, final_download);
        });
      execution_order_.clear();
    }

    while (!execution_order_.empty())
    {
      for (auto it = execution_order_.begin(); it != execution_order_.end();)
      {
        auto* angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[*it].get());
        if (angle_set->IsReady())
        {
          pool_.Run(*it);
          std::swap(*it, execution_order_.back());
          execution_order_.pop_back();
        }
        else
        {
          ++it;
        }
      }
    }
    pool_.WaitAll();

    for (auto& angle_set : angle_agg_)
      static_cast<AAHD_AngleSet*>(angle_set.get())->WaitForDownstreamAndDelayed();

    for (auto& angle_set : angle_agg_)
      angle_set->ResetSweepBuffers();
  };

  auto finalize_saved_angular_flux = [&]()
  {
    for (auto& angle_set : angle_agg_)
    {
      auto* aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_set.get());
      auto* aahd_fluds = static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS());
      aahd_fluds->CopySaveAngularFluxFromDevice();
      aahd_fluds->GetStream().synchronize();
      aahd_fluds->CopySaveAngularFluxToDestinationPsi(
        aah_sweep_chunk.GetProblem(), groupset, *aahd_angle_set);
    }
  };

  if (groupset.fixed_point_sweep)
  {
    const bool use_device_mpi = UseFixedPointSweepDeviceMPI();
    OpenSnInvalidArgumentIf(not use_device_mpi,
                            "Fixed-point GPU sweep requires GPU-aware MPI because delayed "
                            "angular fluxes remain device-resident. Ensure the MPI runtime is "
                            "configured for GPU buffers, e.g. MPICH_GPU_SUPPORT_ENABLED=1.");
    bool final_download_done = false;
    for (unsigned int iter = 0; iter < groupset.fixed_point_sweep_max_iterations; ++iter)
    {
      if (iter > 0)
      {
        aah_sweep_chunk.ZeroDestinationPhi();
        aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
      }
      for (auto& angle_set : angle_agg_)
        static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->ZeroDelayedPsiNewOnDevice();

      final_download_done =
        pullback_angular_flux_on_final_pass and
        iter + 1 == groupset.fixed_point_sweep_max_iterations;
      RecordSweepPass();
      execute_pass(use_device_mpi, use_device_mpi, final_download_done);

      double fixed_point_rel_change = 0.0;
      const bool check_convergence =
        groupset.fixed_point_sweep_tolerance > 0.0 and
        iter + 1 < groupset.fixed_point_sweep_max_iterations;
      if (check_convergence)
      {
        double local_delta_sum = 0.0;
        double local_new_sum = 0.0;
        for (auto& angle_set : angle_agg_)
        {
          const auto [delta_sum, new_sum] =
            static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->ComputeDelayedPsiChangeSumsOnDevice();
          local_delta_sum += delta_sum;
          local_new_sum += new_sum;
        }

        double global_delta_sum = 0.0;
        double global_new_sum = 0.0;
        opensn::mpi_comm.all_reduce(local_delta_sum, global_delta_sum, mpi::op::sum<double>());
        opensn::mpi_comm.all_reduce(local_new_sum, global_new_sum, mpi::op::sum<double>());
        fixed_point_rel_change =
          global_new_sum > 0.0 ? std::sqrt(global_delta_sum / global_new_sum)
                               : std::sqrt(global_delta_sum);
      }

      for (auto& angle_set : angle_agg_)
        static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->CopyDelayedPsiNewToOldOnDevice();
      for (auto& angle_set : angle_agg_)
        static_cast<AAHD_FLUDS*>(&angle_set->GetFLUDS())->GetStream().synchronize();

      if (check_convergence and fixed_point_rel_change < groupset.fixed_point_sweep_tolerance)
      {
        if (pullback_angular_flux_on_final_pass and not final_download_done)
          finalize_saved_angular_flux();
        break;
      }
    }
  }
  else
    execute_pass(false, false, true);

  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, false);
  aah_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();
}

void
SweepScheduler::ScheduleAlgoAsyncFIFO(SweepChunk& sweep_chunk)
{
  // Reset dependency counter
  for (auto& angle_set : angle_agg_)
    angle_set->ResetDependencyCounter();

  auto& cbcd_sweep_chunk = static_cast<CBCDSweepChunk&>(sweep_chunk);
  // Copy phi and source moments to device
  cbcd_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();

  auto& angle_sets = cbcd_sweep_chunk.GetAngleSets();
  auto& fluds_list = cbcd_sweep_chunk.GetFLUDS();
  auto& streams_list = cbcd_sweep_chunk.GetStreams();

  const size_t num_angle_sets = angle_sets.size();
  std::vector<bool> executed(num_angle_sets, 0);
  std::vector<bool> boundary_data_set(num_angle_sets, 0);
  std::vector<bool> kernel_in_flight(num_angle_sets, 0);
  std::vector<std::vector<Task*>> ready_queues(num_angle_sets);
  std::vector<size_t> num_completed_tasks(num_angle_sets, 0);
  std::vector<std::vector<Task*>> ready_tasks(num_angle_sets);
  std::vector<std::vector<std::uint32_t>> ready_cell_ids(num_angle_sets);
  std::vector<std::vector<Task*>> in_flight_tasks(num_angle_sets);
  std::vector<std::vector<std::uint32_t>> in_flight_cell_ids(num_angle_sets);

  for (auto* angle_set : angle_sets)
  {
    auto& current_task_list = angle_set->GetCurrentTaskList();
    if (current_task_list.empty())
      current_task_list = static_cast<const CBC_SPDS&>(angle_set->GetSPDS()).GetTaskList();
  }

  size_t executed_anglesets = 0;
  while (executed_anglesets < num_angle_sets)
  {
    bool any_work_done = false;

    // Poll completed kernels
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (not kernel_in_flight[i])
        continue;
      // Check if the kernel is done
      if (streams_list[i].is_completed())
      {
        // Copy back outgoing (reflecting) boundary and non-local psi
        fluds_list[i]->CopyOutgoingPsiBackToHost(
          cbcd_sweep_chunk, angle_sets[i], in_flight_cell_ids[i]);
        // Update task dependencies
        auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
        for (auto* task : in_flight_tasks[i])
        {
          for (uint64_t succ : task->successors)
          {
            --current_task_list[succ].num_dependencies;
            if (current_task_list[succ].num_dependencies == 0 and boundary_data_set[i])
              ready_queues[i].push_back(&current_task_list[succ]);
          }
          task->completed = true;
        }
        num_completed_tasks[i] += in_flight_tasks[i].size();
        // Send MPI data
        auto* comm = static_cast<CBCD_AsynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
        comm->SendData();
        in_flight_tasks[i].clear();
        in_flight_cell_ids[i].clear();
        kernel_in_flight[i] = false;
        any_work_done = true;
      }
    }

    // Receive and send MPI data
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i])
        continue;
      auto* comm = static_cast<CBCD_AsynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
      auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
      auto received = comm->ReceiveData();
      if (not received.empty())
      {
        for (uint64_t t : received)
        {
          --current_task_list[t].num_dependencies;
          if (current_task_list[t].num_dependencies == 0 and boundary_data_set[i])
            ready_queues[i].push_back(&current_task_list[t]);
        }
        any_work_done = true;
      }
      comm->SendData();
    }

    // Set boundary data
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or boundary_data_set[i] or kernel_in_flight[i])
        continue;
      if (angle_sets[i]->IsDependencyResolved())
      {
        fluds_list[i]->CopyIncomingBoundaryPsiToDevice(cbcd_sweep_chunk, angle_sets[i]);
        boundary_data_set[i] = true;
        any_work_done = true;

        auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
        for (auto& task : current_task_list)
        {
          if (task.num_dependencies == 0 and not task.completed)
            ready_queues[i].push_back(&task);
        }
      }
    }

    // Collect ready tasks and launch kernels (only if task dependencies changed)
    if (any_work_done)
    {
      for (size_t i = 0; i < num_angle_sets; ++i)
      {
        if (executed[i] or (not boundary_data_set[i]) or kernel_in_flight[i])
          continue;

        if (ready_queues[i].empty())
          continue;

        ready_tasks[i] = std::move(ready_queues[i]);
        ready_queues[i].clear();

        ready_cell_ids[i].clear();
        for (auto* task : ready_tasks[i])
          ready_cell_ids[i].push_back(task->reference_id);

        fluds_list[i]->CopyIncomingNonlocalPsiToDevice(angle_sets[i], ready_cell_ids[i]);
        cbcd_sweep_chunk.Sweep(ready_cell_ids[i], i);
        in_flight_tasks[i] = std::move(ready_tasks[i]);
        in_flight_cell_ids[i] = std::move(ready_cell_ids[i]);
        kernel_in_flight[i] = true;
      }
    }

    // Check angleset completion
    for (size_t i = 0; i < num_angle_sets; ++i)
    {
      if (executed[i] or (not boundary_data_set[i]) or kernel_in_flight[i])
        continue;
      auto& current_task_list = angle_sets[i]->GetCurrentTaskList();
      auto* comm = static_cast<CBCD_AsynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
      bool all_done = (num_completed_tasks[i] == current_task_list.size());
      if (all_done and comm->SendData())
      {
        angle_sets[i]->UpdateDependencyCounters();
        executed[i] = true;
        ++executed_anglesets;
        fluds_list[i]->CopySavedPsiFromDevice();
        auto* fluds = fluds_list[i];
        auto* as = angle_sets[i];
        // Cast away constness to add a callback
        streams_list[i].add_callback(
          [fluds, &cbcd_sweep_chunk, as]()
          { fluds->CopySavedPsiToDestinationPsi(cbcd_sweep_chunk, as); });
      }
    }
  }

  // Copy phi and outflow data back to host
  cbcd_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  // Receive delayed data
  opensn::mpi_comm.barrier();
  bool received_delayed_data = false;
  while (not received_delayed_data)
  {
    received_delayed_data = true;

    for (auto& angle_set : angle_sets)
    {
      if (angle_set->FlushSendBuffers() == AngleSetStatus::MESSAGES_PENDING)
        received_delayed_data = false;

      if (not angle_set->ReceiveDelayedData())
        received_delayed_data = false;
    }
  }

  // Reset all
  for (auto& angle_set : angle_sets)
    angle_set->ResetSweepBuffers();
}

} // namespace opensn
