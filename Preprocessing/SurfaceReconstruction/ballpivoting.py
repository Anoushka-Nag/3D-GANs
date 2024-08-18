import open3d as o3d
import numpy as np


mesh = o3d.io.read_triangle_mesh("../../Datasets/airplane_0627.off")
mesh.scale(64 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(50000)
o3d.visualization.draw_geometries([pcd])

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

o3d.visualization.draw_geometries([mesh])