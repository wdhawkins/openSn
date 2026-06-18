// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/device/carrier/carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include <cstdint>
#include <limits>
#include <map>

namespace opensn
{

class SourceXSCarrier : public Carrier
{
public:
  struct BlockData
  {
    const double* scatter = nullptr;
    const double* production = nullptr;
    std::uint32_t num_moments = 0;
    std::uint32_t num_groups = 0;
    bool has_scatter = false;
    bool is_fissionable = false;
  };

  explicit SourceXSCarrier(LBSProblem& lbs_problem);

  BlockData* GetXSGPUData(int block_id);
  const BlockData* GetXSHostData(int block_id) const;

private:
  std::uint64_t ComputeSize(LBSProblem& lbs_problem);
  void Assemble(LBSProblem& lbs_problem);

  std::size_t num_groups_ = std::numeric_limits<std::size_t>::max();
  std::size_t num_moments_ = 0;
  std::map<int, std::uint32_t> block_id_to_index_;
};

} // namespace opensn
