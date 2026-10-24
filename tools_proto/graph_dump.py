"""Write the AAH sweep graph of the tetrahedral cube for offline critical-path modeling.

Same mesh reading (DistributedMeshGenerator), quadrature (14 x 32 GLC, single-angle aggregation),
and therefore the same partition and SPDS as the GPU sweep suite's tet3d runs, but one group and
a CPU AAH sweep, so it runs on any number of ranks. Set OPENSN_SWEEP_GRAPH_DUMP to the output
prefix. Options: mesh_file (required).
"""

if "opensn_console" not in globals():
    from pyopensn.mesh import DistributedMeshGenerator, FromFileMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.source import VolumetricSource
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver

grid = DistributedMeshGenerator(inputs=[FromFileMeshGenerator(filename=mesh_file)]).Execute()
grid.SetUniformBlockID(0)
xs = MultiGroupXS()
xs.CreateSimpleOneGroup(sigma_t=1.0, c=0.5)
pquad = GLCProductQuadrature3DXYZ(n_polar=14, n_azimuthal=32, scattering_order=0)
problem = DiscreteOrdinatesProblem(
    mesh=grid,
    num_groups=1,
    groupsets=[{"groups_from_to": (0, 0), "angular_quadrature": pquad,
                "angle_aggregation_type": "single", "inner_linear_method": "classic_richardson",
                "l_max_its": 1, "l_abs_tol": 1.0e-18}],
    xs_map=[{"block_ids": [0], "xs": xs}],
    volumetric_sources=[VolumetricSource(block_ids=[0], group_strength=[1.0])],
    options={"max_ags_iterations": 1},
    sweep_type="AAH",
)
solver = SteadyStateSourceSolver(problem=problem)
solver.Initialize()
