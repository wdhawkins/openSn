// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/data_types/vector_ghost_communicator/vector_ghost_communicator.h"
#include "framework/runtime.h"
#include "gmock/gmock.h"

using namespace opensn;

TEST(VectorGhostCommunicatorTest, CommunicatesUnevenPartitionsAndCopiesMetadata)
{
  if (mpi_comm.size() != 2)
    GTEST_SKIP() << "Vector ghost communication requires two MPI ranks.";

  const auto rank = mpi_comm.rank();
  const uint64_t local_size = rank == 0 ? 2 : 3;
  const std::vector<uint64_t> ghost_ids =
    rank == 0 ? std::vector<uint64_t>{2, 4} : std::vector<uint64_t>{0};
  VectorGhostCommunicator communicator(local_size, 5, ghost_ids, mpi_comm);
  VectorGhostCommunicator copied(communicator); // NOLINT: The copy is the behavior under test.

  EXPECT_EQ(copied.GetLocalSize(), local_size);
  EXPECT_EQ(copied.GetGlobalSize(), 5);
  EXPECT_EQ(copied.GetGhostIndices(), ghost_ids);

  const std::vector<double> local_values =
    rank == 0 ? std::vector<double>{10.0, 11.0} : std::vector<double>{12.0, 13.0, 14.0};
  auto values = copied.MakeGhostedVector(local_values);
  copied.CommunicateGhostEntries(values);

  if (rank == 0)
    EXPECT_THAT(values, testing::ElementsAre(10.0, 11.0, 12.0, 14.0));
  else
    EXPECT_THAT(values, testing::ElementsAre(12.0, 13.0, 14.0, 10.0));

  for (std::size_t k = 0; k < ghost_ids.size(); ++k)
    EXPECT_EQ(copied.MapGhostToLocal(ghost_ids[k]), local_size + k);
}

TEST(VectorGhostCommunicatorTest, EmptyGhostListsAndInvalidInputs)
{
  if (mpi_comm.size() != 2)
    GTEST_SKIP() << "Vector ghost communication requires two MPI ranks.";

  VectorGhostCommunicator communicator(2, 4, {}, mpi_comm);
  EXPECT_EQ(communicator.MakeGhostedVector(), (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ(communicator.MakeGhostedVector({1.0, 2.0}), (std::vector<double>{1.0, 2.0}));

  auto values = std::vector<double>{1.0, 2.0};
  EXPECT_NO_THROW(communicator.CommunicateGhostEntries(values));
  EXPECT_THROW(communicator.MapGhostToLocal(0), std::invalid_argument);
  EXPECT_THROW(communicator.MakeGhostedVector({1.0}), std::invalid_argument);

  auto wrong_size = std::vector<double>{1.0};
  EXPECT_THROW(communicator.CommunicateGhostEntries(wrong_size), std::invalid_argument);
}

TEST(VectorGhostCommunicatorTest, RejectsOutOfRangeGhostIdentifiers)
{
  if (mpi_comm.size() != 2)
    GTEST_SKIP() << "Vector ghost communication requires two MPI ranks.";

  EXPECT_THROW(VectorGhostCommunicator(2, 4, {4}, mpi_comm), std::invalid_argument);
}
