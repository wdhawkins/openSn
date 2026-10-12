// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/math/unknown_manager/unknown_manager.h"
#include "gtest/gtest.h"

using namespace opensn;

TEST(UnknownManagerTest, MapsMixedUnknownsAndReportsStructureSize)
{
  UnknownManager manager;
  EXPECT_EQ(manager.GetTotalUnknownStructureSize(), 0);

  EXPECT_EQ(manager.AddUnknown(UnknownType::SCALAR), 1);
  EXPECT_EQ(manager.AddUnknown(UnknownType::VECTOR_2), 2);
  EXPECT_EQ(manager.AddUnknown(UnknownType::VECTOR_3), 3);
  EXPECT_EQ(manager.AddUnknown(UnknownType::VECTOR_N, 4), 4);

  EXPECT_EQ(manager.GetNumberOfUnknowns(), 4);
  EXPECT_EQ(manager.GetTotalUnknownStructureSize(), 10);
  EXPECT_EQ(manager.MapUnknown(0), 0);
  EXPECT_EQ(manager.MapUnknown(1, 1), 2);
  EXPECT_EQ(manager.MapUnknown(2, 2), 5);
  EXPECT_EQ(manager.MapUnknown(3, 3), 9);
  EXPECT_EQ(manager.GetUnknown(3).GetNumComponents(), 4);
}

TEST(UnknownManagerTest, NamesConnectionsStorageAndClearAreMutableContracts)
{
  const std::initializer_list<std::pair<UnknownType, unsigned int>> unknowns = {
    {UnknownType::SCALAR, 0}, {UnknownType::VECTOR_N, 3}};
  UnknownManager manager(unknowns, UnknownStorageType::BLOCK);
  EXPECT_EQ(manager.GetDOFStorageType(), UnknownStorageType::BLOCK);

  manager.SetUnknownName(1, "velocity");
  manager.SetUnknownComponentName(1, 2, "z");
  manager.SetUnknownNumOffBlockConnections(1, 4);
  manager.SetUnknownComponentNumOffBlockConnections(1, 2, 7);

  EXPECT_EQ(manager.GetUnknown(1).name, "velocity");
  EXPECT_EQ(manager.GetUnknown(1).component_names[2], "z");
  EXPECT_EQ(manager.GetUnknown(1).num_off_block_connections, (std::vector<int>{4, 4, 7}));

  manager.SetDOFStorageType(UnknownStorageType::NODAL);
  EXPECT_EQ(manager.GetDOFStorageType(), UnknownStorageType::NODAL);
  manager.Clear();
  EXPECT_EQ(manager.GetNumberOfUnknowns(), 0);
  EXPECT_EQ(manager.GetTotalUnknownStructureSize(), 0);
}

TEST(UnknownManagerTest, ConstructingFromUnknownsPreservesNames)
{
  Unknown scalar(UnknownType::SCALAR);
  scalar.name = "temperature";
  scalar.component_names[0] = "value";

  UnknownManager manager({scalar});
  EXPECT_EQ(manager.GetUnknown(0).name, "temperature");
  EXPECT_EQ(manager.GetUnknown(0).component_names[0], "value");

  const auto unitary = UnknownManager::GetUnitaryUnknownManager();
  EXPECT_EQ(unitary.GetNumberOfUnknowns(), 1);
  EXPECT_EQ(unitary.GetTotalUnknownStructureSize(), 1);
}

TEST(UnknownManagerTest, RejectsInvalidUnknownsComponentsAndIdentifiers)
{
  UnknownManager manager;
  EXPECT_THROW(manager.AddUnknown(UnknownType::VECTOR_N, 0), std::runtime_error);
  EXPECT_THROW(manager.AddUnknown(UnknownType::TENSOR, 2), std::runtime_error);
  manager.AddUnknown(UnknownType::VECTOR_2);

  EXPECT_THROW(manager.MapUnknown(1), std::runtime_error);
  EXPECT_THROW(manager.MapUnknown(0, 2), std::out_of_range);
  EXPECT_THROW(manager.SetUnknownName(1, "invalid"), std::runtime_error);
  EXPECT_THROW(manager.SetUnknownComponentName(1, 0, "invalid"), std::runtime_error);
  EXPECT_THROW(manager.SetUnknownComponentName(0, 2, "invalid"), std::runtime_error);
  EXPECT_THROW(manager.SetUnknownNumOffBlockConnections(1, 1), std::runtime_error);
  EXPECT_THROW(manager.SetUnknownComponentNumOffBlockConnections(1, 0, 1), std::runtime_error);
  EXPECT_THROW(manager.SetUnknownComponentNumOffBlockConnections(0, 2, 1), std::runtime_error);
}
