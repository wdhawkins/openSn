// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/scheduler/sweep_scheduler.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/angle_set/aahd_angle_set.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aahd_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include "caliper/cali.h"
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace opensn
{

void
SweepScheduler::ScheduleAlgoAAO(SweepChunk& sweep_chunk)
{
  CALI_CXX_MARK_SCOPE("SweepScheduler::ScheduleAlgoAAO");

  // copy phi and src moments to device
  auto aah_sweep_chunk = static_cast<AAHDSweepChunk&>(sweep_chunk);
  aah_sweep_chunk.GetProblem().CopyPhiAndSrcToDevice();

  std::size_t num_anglesets = angle_agg_.GetNumAngleSets();
  const bool has_reflecting_bc =
    std::any_of(angle_agg_.GetSimBoundaries().begin(),
                angle_agg_.GetSimBoundaries().end(),
                [](const auto& bndry_pair) { return bndry_pair.second->IsReflecting(); });
  static std::atomic<bool> logged_mode{false};
  if (not logged_mode.exchange(true) and mpi_comm.rank() == 0)
    log.Log() << "AAHD AAO mode: "
              << (has_reflecting_bc ? "reflecting-main-compatible" : "nonreflecting-optimized");

  if (has_reflecting_bc)
  {
    // Reflecting path: mirror main branch AAO scheduling.
    std::vector<std::thread> sweep_threads;
    sweep_threads.resize(num_anglesets);

    for (auto& angle_set : angle_agg_)
    {
      auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_set.get());
      aahd_angle_set->SetReflectingCompatibleMode(true);
      aahd_angle_set->SetStartingLatch();
    }

    for (std::size_t i = 0; i < num_anglesets; ++i)
    {
      auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
      sweep_threads[i] =
        std::thread([&sweep_chunk, aahd_angle_set]()
                    { aahd_angle_set->AngleSetAdvance(sweep_chunk, AngleSetStatus::EXECUTE); });
    }

    for (auto& thread : sweep_threads)
      thread.join();

    aah_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

    for (auto& angle_set : angle_agg_)
      angle_set->ResetSweepBuffers();

    return;
  }

  // Initialize (or reinitialize) persistent workers if needed.
  if (aao_workers_initialized_ && aao_worker_threads_.size() != num_anglesets)
  {
    {
      std::scoped_lock<std::mutex> lock(aao_mutex_);
      aao_stop_workers_ = true;
      ++aao_work_epoch_;
    }
    aao_cv_start_.notify_all();
    for (auto& thread : aao_worker_threads_)
      if (thread.joinable())
        thread.join();
    aao_worker_threads_.clear();
    aao_workers_initialized_ = false;
    aao_stop_workers_ = false;
    aao_workers_remaining_ = 0;
    aao_active_sweep_chunk_ = nullptr;
    aao_work_epoch_ = 0;
  }

  if (not aao_workers_initialized_)
  {
    aao_worker_threads_.reserve(num_anglesets);
    for (std::size_t i = 0; i < num_anglesets; ++i)
    {
      aao_worker_threads_.emplace_back(
        [this, i]()
        {
          auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_agg_[i].get());
          std::size_t seen_epoch = 0;
          while (true)
          {
            SweepChunk* active_chunk = nullptr;
            {
              std::unique_lock<std::mutex> lock(aao_mutex_);
              aao_cv_start_.wait(lock,
                                 [this, seen_epoch]()
                                 { return aao_stop_workers_ or aao_work_epoch_ > seen_epoch; });
              if (aao_stop_workers_)
                return;
              seen_epoch = aao_work_epoch_;
              active_chunk = aao_active_sweep_chunk_;
            }

            aahd_angle_set->AngleSetAdvance(*active_chunk, AngleSetStatus::EXECUTE);

            {
              std::scoped_lock<std::mutex> lock(aao_mutex_);
              if (aao_workers_remaining_ > 0)
                --aao_workers_remaining_;
              if (aao_workers_remaining_ == 0)
                aao_cv_done_.notify_one();
            }
          }
        });
    }
    aao_workers_initialized_ = true;
  }

  // set the latches for all anglesets
  for (auto& angle_set : angle_agg_)
  {
    auto aahd_angle_set = static_cast<AAHD_AngleSet*>(angle_set.get());
    aahd_angle_set->SetReflectingCompatibleMode(false);
    aahd_angle_set->SetStartingLatch();
  }

  {
    std::scoped_lock<std::mutex> lock(aao_mutex_);
    aao_active_sweep_chunk_ = &sweep_chunk;
    aao_workers_remaining_ = num_anglesets;
    ++aao_work_epoch_;
  }
  aao_cv_start_.notify_all();

  // wait for all workers to complete this sweep
  {
    std::unique_lock<std::mutex> lock(aao_mutex_);
    aao_cv_done_.wait(lock, [this]() { return aao_workers_remaining_ == 0; });
  }

  // copy phi and outflow data back to host
  aah_sweep_chunk.GetProblem().CopyPhiAndOutflowBackToHost();

  // reset all anglesets
  for (auto& angle_set : angle_agg_)
    angle_set->ResetSweepBuffers();
}

} // namespace opensn
