// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/sweep.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"

namespace opensn
{

void
CommunicateLocationDependencies(const std::vector<int>& location_dependencies,
                                std::vector<std::vector<int>>& global_dependencies)
{

  int P = opensn::mpi_comm.size();

  // Communicate location dep counts
  std::vector<int> depcount_per_loc;
  auto current_loc_dep_count = static_cast<int>(location_dependencies.size());
  mpi_comm.all_gather(current_loc_dep_count, depcount_per_loc);

  // Broadcast dependencies
  std::vector<int> raw_depvec_displs(P, 0);
  for (int locI = 1; locI < P; ++locI)
    raw_depvec_displs[locI] = raw_depvec_displs[locI - 1] + depcount_per_loc[locI - 1];

  std::vector<int> raw_dependencies;
  mpi_comm.all_gather(location_dependencies, raw_dependencies, depcount_per_loc, raw_depvec_displs);

  for (int locI = 0; locI < P; ++locI)
  {
    global_dependencies[locI].resize(depcount_per_loc[locI], 0);
    for (int c = 0; c < depcount_per_loc[locI]; ++c)
    {
      int addr = raw_depvec_displs[locI] + c;
      global_dependencies[locI][c] = raw_dependencies[addr];
    }
  }
}

std::vector<std::vector<std::vector<int>>>
BatchCommunicateLocationDependencies(const std::vector<std::vector<int>>& local_deps)
{
  const int P = opensn::mpi_comm.size();
  const int N = static_cast<int>(local_deps.size());

  std::vector<std::vector<std::vector<int>>> result(N, std::vector<std::vector<int>>(P));

  if (N == 0)
    return result;

  // All-gather dep counts: each rank sends N ints, all receive P*N ints.
  std::vector<int> local_counts(N);
  for (int s = 0; s < N; ++s)
    local_counts[s] = static_cast<int>(local_deps[s].size());

  std::vector<int> all_counts(P * N);
  mpi_comm.all_gather(local_counts.data(), N, all_counts.data(), N);

  // Build flat send buffer (concatenate all local dep vectors).
  std::vector<int> flat_deps;
  {
    int total = 0;
    for (int s = 0; s < N; ++s)
      total += local_counts[s];
    flat_deps.reserve(total);
    for (int s = 0; s < N; ++s)
      flat_deps.insert(flat_deps.end(), local_deps[s].begin(), local_deps[s].end());
  }

  // Compute per-rank receive counts and displacements for all_gatherv.
  std::vector<int> recv_counts(P), recv_displs(P, 0);
  for (int r = 0; r < P; ++r)
  {
    int sz = 0;
    for (int s = 0; s < N; ++s)
      sz += all_counts[r * N + s];
    recv_counts[r] = sz;
  }
  for (int r = 1; r < P; ++r)
    recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];

  const int total_recv = recv_displs[P - 1] + recv_counts[P - 1];
  std::vector<int> all_flat_deps(total_recv);
  mpi_comm.all_gather(flat_deps, all_flat_deps, recv_counts, recv_displs);

  // Unpack: result[s][r] = deps for SPDS s on rank r.
  for (int r = 0; r < P; ++r)
  {
    int offset = recv_displs[r];
    for (int s = 0; s < N; ++s)
    {
      const int cnt = all_counts[r * N + s];
      result[s][r].assign(all_flat_deps.begin() + offset, all_flat_deps.begin() + offset + cnt);
      offset += cnt;
    }
  }

  return result;
}

} // namespace opensn
