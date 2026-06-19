// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/outflow/outflow_carrier.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_view.h"
#include <algorithm>

namespace opensn
{

namespace
{

constexpr unsigned int OutflowZeroBlockSize = 256;

__CRB_GLOBAL_FUNC__ void
ZeroOutflowGroupsKernel(double* outflow,
                        std::size_t size,
                        unsigned int first_group,
                        unsigned int num_groups_in_groupset,
                        unsigned int total_num_groups)
{
#if defined(__NVCC__) || defined(__HIPCC__)
  std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  auto work_index = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  std::size_t i = work_index.get_global_id(2);
  const std::size_t stride = work_index.get_local_range(2) * work_index.get_group_range(2);
#endif

  while (i < size)
  {
    const auto group = static_cast<unsigned int>(i % total_num_groups);
    if (group >= first_group and group < first_group + num_groups_in_groupset)
      outflow[i] = 0.0;
    i += stride;
  }
}

} // namespace

OutflowCarrier::OutflowCarrier(LBSProblem& lbs_problem) : bank_(lbs_problem.GetOutflowBank())
{
  if (bank_.GetSize() > 0)
    device_outflows_ = crb::DeviceMemory<double>(bank_.GetSize());
}

void
OutflowCarrier::CopyToDevice()
{
  const auto size = bank_.GetSize();
  if (size == 0)
    return;

  auto& host_outflows = bank_.GetOutflowData();
  crb::copy(device_outflows_, host_outflows, size);
}

void
OutflowCarrier::CopyFromDevice()
{
  const auto size = bank_.GetSize();
  if (size == 0)
    return;

  auto& host_outflows = bank_.GetOutflowData();
  crb::copy(host_outflows, device_outflows_, size);
}

void
OutflowCarrier::ZeroGroupsOnDevice(unsigned int first_group,
                                   unsigned int num_groups_in_groupset,
                                   unsigned int total_num_groups)
{
  const auto size = bank_.GetSize();
  if (size == 0)
    return;

  const auto num_blocks = static_cast<unsigned int>(
    std::min<std::size_t>((size + OutflowZeroBlockSize - 1) / OutflowZeroBlockSize, 65535));

#if defined(__NVCC__) || defined(__HIPCC__)
  ZeroOutflowGroupsKernel<<<num_blocks, OutflowZeroBlockSize>>>(device_outflows_.get(),
                                                                 size,
                                                                 first_group,
                                                                 num_groups_in_groupset,
                                                                 total_num_groups);
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
  crb::Stream stream;
  crb::Dim3 block(OutflowZeroBlockSize);
  crb::Dim3 grid(num_blocks);
  stream.parallel_for(sycl::nd_range<3>(grid * block, block),
                      [=](sycl::nd_item<3>)
                      {
                        ZeroOutflowGroupsKernel(device_outflows_.get(),
                                                size,
                                                first_group,
                                                num_groups_in_groupset,
                                                total_num_groups);
                      });
  stream.synchronize();
#endif
}

std::uint64_t
OutflowCarrier::GetOffset(const std::uint32_t& cell_local_idx, const std::uint32_t& face_idx)
{
  return bank_.GetOffset(cell_local_idx, face_idx);
}

bool
OutflowCarrier::HasOffset(const std::uint32_t& cell_local_idx, const std::uint32_t& face_idx)
{
  return bank_.HasOffset(cell_local_idx, face_idx);
}

} // namespace opensn
