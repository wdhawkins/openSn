#include "framework/mesh/surface_mesh/surface_mesh.h"
#include <gtest/gtest.h>
#include <filesystem>

using namespace opensn;
namespace fs = std::filesystem;

namespace
{

auto meshes_loc = fs::path(OPENSN_TEST_ROOT) / "assets" / "mesh";

class SurfaceMeshProbe : public SurfaceMesh
{
public:
  std::size_t GetPolygonCount() const { return poly_faces_.size(); }
  const PolyFace& GetPolygon(const std::size_t index) const { return *poly_faces_.at(index); }
  const std::vector<int>& GetPhysicalRegionMap() const { return physical_region_map_; }
};

} // namespace

TEST(SurfaceMeshTest, ImportOBJ)
{
  SurfaceMesh mesh;
  EXPECT_EQ(mesh.ImportFromOBJFile(meshes_loc / "test_surface1.obj"), 0);
  EXPECT_EQ(mesh.GetVertices().size(), 8);
  EXPECT_EQ(mesh.GetTriangles().size(), 6);

  mesh.UpdateInternalConnectivity();
}

TEST(SurfaceMeshTest, ImportOBJMissingFile)
{
  SurfaceMesh mesh;
  EXPECT_THROW(mesh.ImportFromOBJFile(meshes_loc / "does_not_exist.obj"), std::runtime_error);
}

TEST(SurfaceMeshTest, ImportOBJAsPolygonsAppliesTranslationAndBuildsConnectivity)
{
  SurfaceMeshProbe mesh;
  const Vector3 translation(1.0, -2.0, 3.0);
  ASSERT_EQ(mesh.ImportFromOBJFile(meshes_loc / "test_surface1.obj", true, translation), 0);

  ASSERT_EQ(mesh.GetVertices().size(), 8);
  EXPECT_TRUE(mesh.GetVertices().front().AbsoluteEquals(Vector3(-3.0, -6.0, 3.0), 1.0e-14));
  EXPECT_EQ(mesh.GetTriangles().size(), 0);
  EXPECT_EQ(mesh.GetPolygonCount(), 6);

  bool found_connected_edge = false;
  for (std::size_t f = 0; f < mesh.GetPolygonCount(); ++f)
    for (const auto& edge : mesh.GetPolygon(f).edges)
      found_connected_edge = found_connected_edge or edge[2] >= 0;
  EXPECT_TRUE(found_connected_edge);
}

TEST(SurfaceMeshTest, ImportTriangleFilesBuildsPolygonGeometry)
{
  SurfaceMeshProbe mesh;
  const auto base_name = (meshes_loc / "surface_triangle").string();
  ASSERT_EQ(mesh.ImportFromTriangleFiles(base_name.c_str(), true), 0);

  ASSERT_EQ(mesh.GetVertices().size(), 4);
  ASSERT_EQ(mesh.GetPolygonCount(), 2);
  EXPECT_TRUE(
    mesh.GetPolygon(0).face_centroid.AbsoluteEquals(Vector3(2.0 / 3.0, 1.0 / 3.0, 0.0), 1.0e-14));
  EXPECT_TRUE(
    mesh.GetPolygon(1).face_centroid.AbsoluteEquals(Vector3(1.0 / 3.0, 2.0 / 3.0, 0.0), 1.0e-14));
  EXPECT_NEAR(mesh.GetPolygon(0).geometric_normal.Norm(), 1.0, 1.0e-14);
  EXPECT_NEAR(mesh.GetPolygon(1).geometric_normal.Norm(), 1.0, 1.0e-14);

  std::vector<double> x_cuts = {0.5};
  std::vector<double> y_cuts = {0.5};
  EXPECT_NO_THROW(mesh.ComputeLoadBalancing(x_cuts, y_cuts));
}

TEST(SurfaceMeshTest, ImportGmshV2ReadsTrianglesQuadsAndPhysicalRegions)
{
  SurfaceMeshProbe mesh;
  const auto file_name = (meshes_loc / "surface_mixed_v2.msh").string();
  ASSERT_EQ(mesh.ImportFromMshFiles(file_name.c_str(), true), 0);

  EXPECT_EQ(mesh.GetVertices().size(), 5);
  ASSERT_EQ(mesh.GetPolygonCount(), 2);
  EXPECT_EQ(mesh.GetPolygon(0).v_indices.size(), 4);
  EXPECT_EQ(mesh.GetPolygon(1).v_indices.size(), 3);
  EXPECT_EQ(mesh.GetPhysicalRegionMap(), (std::vector<int>{7, 8}));

  bool polygons_are_neighbors = false;
  for (const auto& edge : mesh.GetPolygon(0).edges)
    polygons_are_neighbors = polygons_are_neighbors or edge[2] == 1;
  EXPECT_TRUE(polygons_are_neighbors);
}

TEST(SurfaceMeshTest, MissingTriangleAndGmshFilesAreRejected)
{
  SurfaceMesh mesh;
  const auto missing_triangle = (meshes_loc / "does_not_exist").string();
  const auto missing_gmsh = (meshes_loc / "does_not_exist.msh").string();
  EXPECT_THROW(mesh.ImportFromTriangleFiles(missing_triangle.c_str(), true), std::runtime_error);
  EXPECT_THROW(mesh.ImportFromMshFiles(missing_gmsh.c_str(), true), std::runtime_error);
}
