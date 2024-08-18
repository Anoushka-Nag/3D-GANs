import open3d as o3d
import numpy as np


mesh = o3d.io.read_triangle_mesh("../../Datasets/airplane_0627.off")
mesh.scale(63 / (np.max(mesh.get_max_bound() - mesh.get_min_bound())),
           center=mesh.get_center())
mesh.compute_vertex_normals()

voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=1)

np_array = np.zeros((64, 64, 64))

for voxel in voxel_grid.get_voxels():
    grid = voxel.grid_index
    np_array[grid[0], grid[1], grid[2]] = 1


voxel_grid = o3d.geometry.VoxelGrid()

idx = np.where(np_array == 1)

