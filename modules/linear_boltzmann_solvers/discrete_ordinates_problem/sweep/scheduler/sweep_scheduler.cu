// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/spds/cbc.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/communicators/cbcd_async_comm.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/cbcd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caribou/main.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <array>
#include <thread>
#include <vector>

namespace opensn
{

namespace
{

/// Prototype settings of the fused scheduler, read once from the environment (experiments only).
struct FusedKnobs
{
  /// OPENSN_FUSED_EARLY: finish each angle set when its own part of a run is done.
  bool early = false;
  /// OPENSN_FUSED_SPREAD: split the ready angle sets evenly over the idle lanes.
  bool spread = false;
  /// OPENSN_FUSED_TIMELINE: print per-rank scheduler timing for every sweep.
  bool timeline = false;
  /// OPENSN_FUSED_MAXBATCH: maximum angle sets per run (default 256).
  std::size_t max_batch = 256;
  /// OPENSN_FUSED_ROLLING: sweep every active angle set's next level in each launch.
  bool rolling = false;
  /// OPENSN_FUSED_AHEAD: rolling steps the host may submit before the device finishes them.
  std::size_t ahead = 2;
  /// OPENSN_FUSED_RSTREAMS: number of rolling streams.
  std::size_t rolling_streams = 1;
};

const FusedKnobs&
GetFusedKnobs()
{
  static const FusedKnobs knobs = []
  {
    auto flag = [](const char* name)
    {
      const char* v = std::getenv(name);
      return v != nullptr and std::atoi(v) != 0;
    };
    FusedKnobs k;
    k.early = flag("OPENSN_FUSED_EARLY");
    k.spread = flag("OPENSN_FUSED_SPREAD");
    k.timeline = flag("OPENSN_FUSED_TIMELINE");
    k.rolling = flag("OPENSN_FUSED_ROLLING");
    if (const char* v = std::getenv("OPENSN_FUSED_RSTREAMS"))
      k.rolling_streams = std::max<std::size_t>(std::strtoul(v, nullptr, 10), 1);
    if (const char* v = std::getenv("OPENSN_FUSED_AHEAD"))
      k.ahead = std::max<std::size_t>(std::strtoul(v, nullptr, 10), 1);
    if (const char* v = std::getenv("OPENSN_FUSED_MAXBATCH"))
      k.max_batch = std::clamp<std::size_t>(std::strtoul(v, nullptr, 10), 1, 256);
    return k;
  }();
  return knobs;
}

} // namespace

void
SweepScheduler::ScheduleAlgoAAO(SweepChunk& sweep_chunk)
{
  // copy phi and src moments to device
  auto& aah_sweep_chunk = dynamic_cast<AAHDSweepChunk&>(sweep_chunk);
  int groupset_id = angle_agg_.GetGroupsetID();
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, true);

  // reset dependency counters and pre-post receives for all anglesets
  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();
  execution_order_.resize(num_anglesets);
  for (std::size_t i = 0; i < num_anglesets; ++i)
  {
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_agg_[i].get());
    aahd_angle_set->ResetDependencyCounter();
    aahd_angle_set->PrepostReceives();
    execution_order_[i] = i;
  }

  // assign sweep task to thread pool (but not execution yet)
  pool_.AssignTask(
    [this, &sweep_chunk](std::size_t i)
    {
      auto* aahd = dynamic_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      aahd->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE);
    });

  // poll for readiness and launch threads
  while (!execution_order_.empty())
  {
    for (auto it = execution_order_.begin(); it != execution_order_.end();)
    {
      auto* angle_set = dynamic_cast<AAHD_AngleSet*>(angle_agg_[*it].get());
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

  // wait for sends and receive of delayed data
  for (auto& angle_set : angle_agg_)
  {
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_set.get());
    aahd_angle_set->WaitForDownstreamAndDelayed();
  }

  // copy boundary, phi and outflow data back to host
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, false);
  aah_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  // reset all anglesets
  for (auto& angle_set : angle_agg_)
    angle_set->ResetSweepBuffers();
}

void
SweepScheduler::ScheduleAlgoAAOFused(SweepChunk& sweep_chunk)
{
  auto& aah_sweep_chunk = dynamic_cast<AAHDSweepChunk&>(sweep_chunk);
  int groupset_id = angle_agg_.GetGroupsetID();
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, true);

  // Every lane's stream is idle here. The delayed-psi uploads in PrepostReceives go on lane 0. A
  // run may use another lane, so lane 0 is synchronized before any run starts.
  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();
  std::vector<AAHD_AngleSet*> all_anglesets(num_anglesets);
  crb::Stream& setup_stream = aah_sweep_chunk.GetLaneStream(0);
  for (std::size_t i = 0; i < num_anglesets; ++i)
  {
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_agg_[i].get());
    aahd_angle_set->ResetDependencyCounter();
    aahd_angle_set->SetFLUDSStream(setup_stream);
    aahd_angle_set->PrepostReceives();
    all_anglesets[i] = aahd_angle_set;
  }
  setup_stream.synchronize();

  const FusedKnobs& knobs = GetFusedKnobs();
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  double t_idle = 0.0, t_finish = 0.0, t_submit = 0.0, t_poll = 0.0;
  std::size_t num_runs = 0, num_members = 0, widest = 0, num_launches = 0;
  auto seconds = [](Clock::time_point a, Clock::time_point b)
  { return std::chrono::duration<double>(b - a).count(); };

  // Maximum angle sets per fused run. Each member is one grid-z slice, so this bounds the launch
  // well below the CUDA/HIP grid-z limit (65535) and bounds the per-run slot array at
  // max_batch_width * levels entries.
  const std::size_t max_batch_width = knobs.max_batch;
  const std::size_t num_lanes = aah_sweep_chunk.NumFusedLanes();
  std::vector<std::vector<AAHD_AngleSet*>> lane_runs(num_lanes);
  // Angle sets that have not started, in angle-set order. Only these are polled for readiness.
  std::vector<AAHD_AngleSet*> pending = all_anglesets;
  for (auto* as : all_anglesets)
    as->CompletionReady().store(false, std::memory_order_relaxed);
  std::size_t num_finished = 0;
  std::vector<std::size_t> free_lanes;
  std::vector<AAHD_AngleSet*> ready;

  // Prototype trace: per angle set, when it was first seen ready, submitted, and finished, on the
  // node-wide steady clock (ns), for the fourth sweep only.
  static int sweep_count = 0;
  ++sweep_count;
  const char* trace_prefix = std::getenv("OPENSN_FUSED_TRACE");
  const bool trace = trace_prefix != nullptr and sweep_count == 4;
  std::map<std::size_t, std::array<long long, 3>> trace_times;
  auto now_ns = []
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  };

  // Rolling runs: each launch sweeps the next level of every active angle set on lane 0. Angle
  // sets that become ready join at level 0 in the next launch, and finished ones leave.
  if (knobs.rolling)
  {
    // Each rolling stream has its own active angle sets and ring of in-flight steps. A newly
    // ready angle set joins the stream with the fewest active angle sets.
    struct RollingState
    {
      std::vector<AAHDSweepChunk::RollingMember> active;
      std::vector<std::pair<std::size_t, std::vector<AAHD_AngleSet*>>> inflight;
      std::size_t first_inflight = 0;
      std::size_t step = 0;
    };
    const std::size_t num_streams = std::min(knobs.rolling_streams, num_lanes);
    std::vector<RollingState> streams(num_streams);
    aah_sweep_chunk.InitRolling(num_streams, knobs.ahead, num_anglesets);
    std::vector<AAHD_AngleSet*> completing;
    while (num_finished < num_anglesets)
    {
      auto t0 = Clock::now();
      for (std::size_t k = 0; k < num_streams; ++k)
      {
        auto& st = streams[k];
        while (st.first_inflight < st.inflight.size() and
               aah_sweep_chunk.IsRollingStepDone(k, st.inflight[st.first_inflight].first))
        {
          for (auto* as : st.inflight[st.first_inflight].second)
          {
            as->FinishCompletion(sweep_chunk);
            ++num_finished;
          }
          ++st.first_inflight;
        }
      }
      auto t1 = Clock::now();
      t_finish += seconds(t0, t1);

      // Keep the device fed first. Scan for newly ready angle sets only when no stream can
      // submit a step, because each scan tests every pending angle set's receives.
      bool submitted = false, any_inflight = false, any_active = false;
      for (std::size_t k = 0; k < num_streams; ++k)
      {
        auto& st = streams[k];
        const std::size_t num_inflight = st.inflight.size() - st.first_inflight;
        if (not st.active.empty() and num_inflight < knobs.ahead)
        {
          widest = std::max(widest, st.active.size());
          num_launches +=
            aah_sweep_chunk.SubmitRollingStep(k, st.step % knobs.ahead, st.active, completing);
          st.inflight.emplace_back(st.step % knobs.ahead, completing);
          ++st.step;
          ++num_runs;
          submitted = true;
        }
        any_inflight = any_inflight or st.inflight.size() > st.first_inflight;
        any_active = any_active or not st.active.empty();
      }
      auto t2 = Clock::now();
      t_submit += seconds(t1, t2);
      if (submitted)
        continue;

      auto still_pending = pending.begin();
      for (auto* as : pending)
        if (as->IsReady())
        {
          std::size_t k = 0;
          for (std::size_t j = 1; j < num_streams; ++j)
            if (streams[j].active.size() < streams[k].active.size())
              k = j;
          as->PrepareForBatch(sweep_chunk, aah_sweep_chunk.GetLaneStream(k));
          AAHDSweepChunk::RollingMember m{as, 0, {}};
          if (as->NumLevels() > 0)
            as->FillLevelSlot(0, m.slot);
          streams[k].active.push_back(m);
          ++num_members;
          any_active = true;
        }
        else
          *still_pending++ = as;
      pending.erase(still_pending, pending.end());
      auto t3 = Clock::now();
      t_poll += seconds(t2, t3);
      if (not any_inflight and not any_active and not pending.empty())
        t_idle += seconds(t0, t3);
    }
  }

  // Runs are pipelined. Up to num_lanes runs are in flight, each on its own stream. Finished
  // members are completed, which releases their followers and sends their downstream angular
  // fluxes. The angle sets that are ready then start on a free lane. They do not wait for the
  // other runs. A run's members are fixed when it starts.
  while (num_finished < num_anglesets)
  {
    auto t0 = Clock::now();
    bool in_flight = false;
    for (std::size_t lane = 0; lane < num_lanes; ++lane)
    {
      auto& run = lane_runs[lane];
      if (run.empty())
        continue;
      if (knobs.early)
      {
        auto keep = run.begin();
        for (auto* as : run)
          if (as->CompletionReady().load(std::memory_order_acquire))
          {
            if (trace)
              trace_times[as->GetID()][2] = now_ns();
            as->FinishCompletion(sweep_chunk);
            ++num_finished;
          }
          else
            *keep++ = as;
        run.erase(keep, run.end());
      }
      else if (aah_sweep_chunk.IsLaneIdle(lane))
      {
        for (auto* as : run)
        {
          if (trace)
            trace_times[as->GetID()][2] = now_ns();
          as->FinishCompletion(sweep_chunk);
          ++num_finished;
        }
        run.clear();
      }
      in_flight = in_flight or not run.empty();
    }
    auto t1 = Clock::now();
    t_finish += seconds(t0, t1);

    // A lane is free when its run has no unfinished members and its stream is idle.
    free_lanes.clear();
    for (std::size_t lane = 0; lane < num_lanes; ++lane)
      if (lane_runs[lane].empty() and (not knobs.early or aah_sweep_chunk.IsLaneIdle(lane)))
        free_lanes.push_back(lane);
      else
        in_flight = true;
    ready.clear();
    if (not free_lanes.empty())
    {
      const std::size_t capacity = free_lanes.size() * max_batch_width;
      auto still_pending = pending.begin();
      for (auto* as : pending)
        if (ready.size() < capacity and as->IsReady())
        {
          if (trace)
            trace_times[as->GetID()][0] = now_ns();
          ready.push_back(as);
        }
        else
          *still_pending++ = as;
      pending.erase(still_pending, pending.end());
    }
    auto t2 = Clock::now();
    t_poll += seconds(t1, t2);
    if (ready.empty())
    {
      if (not in_flight and not pending.empty())
        t_idle += seconds(t0, t2);
      continue;
    }

    // Split the ready angle sets over the free lanes: evenly with spread, otherwise the first
    // lane takes up to a full run.
    std::size_t per_lane = max_batch_width;
    if (knobs.spread)
      per_lane =
        std::min(max_batch_width, (ready.size() + free_lanes.size() - 1) / free_lanes.size());
    std::size_t next = 0;
    for (std::size_t lane : free_lanes)
    {
      if (next == ready.size())
        break;
      const std::size_t count = std::min(per_lane, ready.size() - next);
      auto& run = lane_runs[lane];
      crb::Stream& stream = aah_sweep_chunk.GetLaneStream(lane);
      for (std::size_t i = next; i < next + count; ++i)
      {
        if (trace)
          trace_times[ready[i]->GetID()][1] = now_ns();
        ready[i]->PrepareForBatch(sweep_chunk, stream);
        run.push_back(ready[i]);
      }
      next += count;
      num_launches += aah_sweep_chunk.SubmitBatchRun(lane, run, knobs.early);
      ++num_runs;
      num_members += count;
      widest = std::max(widest, count);
    }
    t_submit += seconds(t2, Clock::now());
  }

  if (trace)
  {
    std::ofstream out(std::string(trace_prefix) + ".r" + std::to_string(opensn::mpi_comm.rank()) +
                      ".txt");
    for (const auto& [id, t] : trace_times)
      out << id << " " << t[0] << " " << t[1] << " " << t[2] << "\n";
  }

  if (knobs.timeline)
    log.LogAll() << "FUSED_TL rank " << opensn::mpi_comm.rank() << " sweep_ms "
                 << 1e3 * seconds(t_start, Clock::now()) << " idle_ms " << 1e3 * t_idle
                 << " finish_ms " << 1e3 * t_finish << " poll_ms " << 1e3 * t_poll << " submit_ms "
                 << 1e3 * t_submit << " runs " << num_runs << " avg_width "
                 << (num_runs ? static_cast<double>(num_members) / num_runs : 0.0) << " max_width "
                 << widest << " launches " << num_launches;

  // Wait for sends and receive of delayed data
  for (auto& angle_set : angle_agg_)
  {
    auto* aahd_angle_set = dynamic_cast<AAHD_AngleSet*>(angle_set.get());
    aahd_angle_set->WaitForDownstreamAndDelayed();
  }

  // Copy boundary, phi and outflow data back to host
  aah_sweep_chunk.GetProblem().TransferDeviceBoundaryData(groupset_id, false);
  aah_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  // Reset all angle sets
  for (auto& angle_set : angle_agg_)
    angle_set->ResetSweepBuffers();
}

void
SweepScheduler::ScheduleAlgoAsyncFIFO(SweepChunk& sweep_chunk)
{
  // Reset dependency counter
  for (auto& angle_set : angle_agg_)
    angle_set->ResetDependencyCounter();

  auto& cbcd_sweep_chunk = dynamic_cast<CBCDSweepChunk&>(sweep_chunk);
  // Copy phi and source moments to device
  cbcd_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();

  const auto& angle_sets = cbcd_sweep_chunk.GetAngleSets();
  const auto& fluds_list = cbcd_sweep_chunk.GetFLUDS();
  auto& streams_list = cbcd_sweep_chunk.GetStreams();

  const size_t num_angle_sets = angle_sets.size();
  std::vector<bool> executed(num_angle_sets, false);
  std::vector<bool> boundary_data_set(num_angle_sets, false);
  std::vector<bool> kernel_in_flight(num_angle_sets, false);
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
      current_task_list = dynamic_cast<const CBC_SPDS&>(angle_set->GetSPDS()).GetTaskList();
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
        auto* comm = dynamic_cast<CBCD_AsynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
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
      auto* comm = dynamic_cast<CBCD_AsynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
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
      auto* comm = dynamic_cast<CBCD_AsynchronousCommunicator*>(angle_sets[i]->GetCommunicator());
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

    for (const auto& angle_set : angle_sets)
    {
      if (angle_set->FlushSendBuffers() == AngleSetStatus::MESSAGES_PENDING)
        received_delayed_data = false;

      if (not angle_set->ReceiveDelayedData())
        received_delayed_data = false;
    }
  }

  // Reset all
  for (const auto& angle_set : angle_sets)
    angle_set->ResetSweepBuffers();
}

} // namespace opensn
