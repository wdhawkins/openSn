// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/graphs/linear_graph_partitioner.h"
#include "framework/mesh/mesh_generator/orthogonal_mesh_generator.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/parameters/parameter_block.h"
#include "framework/runtime.h"
#include "gtest/gtest.h"

using namespace opensn;

namespace
{

std::shared_ptr<MeshContinuum>
BuildOrthogonalMesh(const std::vector<std::vector<double>>& node_sets,
                    const bool distributed_generation)
{
  ParameterBlock node_sets_array("node_sets");
  for (std::size_t d = 0; d < node_sets.size(); ++d)
    node_sets_array.AddParameter(ParameterBlock(std::to_string(d), node_sets[d]));
  node_sets_array.ChangeToArray();

  auto partitioner_params = LinearGraphPartitioner::GetInputParameters();
  auto partitioner = std::make_shared<LinearGraphPartitioner>(partitioner_params);

  ParameterBlock block;
  block.AddParameter(node_sets_array);
  block.AddParameter("distributed_generation", distributed_generation);
  block.AddParameter("partitioner", std::shared_ptr<GraphPartitioner>(partitioner));

  auto params = OrthogonalMeshGenerator::GetInputParameters();
  params.AssignParameters(block);
  return OrthogonalMeshGenerator(params).Execute();
}

void
ExpectEquivalentLocalMeshes(const MeshContinuum& expected, const MeshContinuum& actual)
{
  EXPECT_EQ(actual.GetDimension(), expected.GetDimension());
  EXPECT_EQ(actual.GetType(), expected.GetType());
  EXPECT_EQ(actual.GetCoordinateSystem(), expected.GetCoordinateSystem());
  EXPECT_EQ(actual.GetGlobalNumberOfCells(), expected.GetGlobalNumberOfCells());
  EXPECT_EQ(actual.GetGlobalVertexCount(), expected.GetGlobalVertexCount());
  EXPECT_EQ(actual.GetLocalCellCount(), expected.GetLocalCellCount());
  EXPECT_EQ(actual.GhostCellCount(), expected.GhostCellCount());
  EXPECT_EQ(actual.GetIJKInfo(), expected.GetIJKInfo());
  EXPECT_EQ(actual.GetBoundaryIDMap(), expected.GetBoundaryIDMap());

  for (const auto& expected_cell : expected.GetLocalCells())
  {
    ASSERT_TRUE(actual.IsCellLocal(expected_cell->global_id));
    const auto& actual_cell = actual.GetGlobalCell(expected_cell->global_id);

    EXPECT_EQ(actual_cell.GetType(), expected_cell->GetType());
    EXPECT_EQ(actual_cell.GetSubType(), expected_cell->GetSubType());
    EXPECT_EQ(actual_cell.partition_id, expected_cell->partition_id);
    EXPECT_EQ(actual_cell.vertex_ids, expected_cell->vertex_ids);
    EXPECT_TRUE(actual_cell.centroid.AbsoluteEquals(expected_cell->centroid, 1.0e-14));
    EXPECT_NEAR(actual_cell.volume, expected_cell->volume, 1.0e-14);
    ASSERT_EQ(actual_cell.faces.size(), expected_cell->faces.size());

    for (std::size_t f = 0; f < expected_cell->faces.size(); ++f)
    {
      const auto& expected_face = expected_cell->faces[f];
      const auto& actual_face = actual_cell.faces[f];
      EXPECT_EQ(actual_face.has_neighbor, expected_face.has_neighbor);
      EXPECT_EQ(actual_face.neighbor_id, expected_face.neighbor_id);
      EXPECT_EQ(actual_face.vertex_ids, expected_face.vertex_ids);
      EXPECT_TRUE(actual_face.centroid.AbsoluteEquals(expected_face.centroid, 1.0e-14));
      EXPECT_TRUE(actual_face.normal.AbsoluteEquals(expected_face.normal, 1.0e-14));
      EXPECT_NEAR(actual_face.area, expected_face.area, 1.0e-14);
    }
  }
}

} // namespace

TEST(OrthogonalMeshGeneratorTest, DistributedGenerationMatchesStandardMesh)
{
  if (mpi_comm.size() != 2)
    GTEST_SKIP() << "Distributed generation comparison requires two MPI ranks.";

  const std::vector<std::vector<std::vector<double>>> node_sets_by_dimension = {
    {{-1.0, -0.25, 0.5, 2.0}},
    {{-1.0, 0.0, 2.0}, {-2.0, -0.5, 1.0}},
    {{-1.0, 0.0, 2.0}, {-2.0, -0.5, 1.0}, {0.0, 0.25, 1.5}},
  };

  for (const auto& node_sets : node_sets_by_dimension)
  {
    const auto standard_grid = BuildOrthogonalMesh(node_sets, false);
    const auto distributed_grid = BuildOrthogonalMesh(node_sets, true);
    ExpectEquivalentLocalMeshes(*standard_grid, *distributed_grid);
  }
}
