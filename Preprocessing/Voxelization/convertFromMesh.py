import open3d as o3d
import numpy as np


mesh = o3d.io.read_triangle_mesh("../../Datasets/car_0004.off")
mesh.scale(128 / (np.max(mesh.get_max_bound() - mesh.get_min_bound())),
           center=mesh.get_center())
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=1)
o3d.visualization.draw_geometries([voxel_grid])
