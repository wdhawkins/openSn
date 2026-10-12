// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/graphs/linear_graph_partitioner.h"
#include "gmock/gmock.h"

using namespace opensn;

namespace
{

LinearGraphPartitioner
MakePartitioner(const int all_to_rank = -1)
{
  auto params = LinearGraphPartitioner::GetInputParameters();
  ParameterBlock supplied;
  supplied.AddParameter("all_to_rank", all_to_rank);
  params.AssignParameters(supplied);
  return LinearGraphPartitioner(params);
}

} // namespace

TEST(LinearGraphPartitionerTest, DistributesContiguousUnevenSubsets)
{
  auto partitioner = MakePartitioner();
  const std::vector<std::vector<uint64_t>> graph(7);
  EXPECT_THAT(partitioner.Partition(graph, {}, 3), testing::ElementsAre(0, 0, 0, 1, 1, 2, 2));
}

TEST(LinearGraphPartitionerTest, SupportsMorePartitionsThanItemsAndEmptyGraphs)
{
  auto partitioner = MakePartitioner();
  EXPECT_THAT(partitioner.Partition(std::vector<std::vector<uint64_t>>(2), {}, 4),
              testing::ElementsAre(0, 1));
  EXPECT_TRUE(partitioner.Partition({}, {}, 3).empty());
}

TEST(LinearGraphPartitionerTest, CanAssignEveryItemToOneRank)
{
  auto partitioner = MakePartitioner(2);
  EXPECT_THAT(partitioner.Partition(std::vector<std::vector<uint64_t>>(5), {}, 4),
              testing::ElementsAre(2, 2, 2, 2, 2));
}
