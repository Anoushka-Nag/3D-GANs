import open3d as o3d
import numpy as np
from utils import data
from utils.visualization import plot_3d_voxels


mesh = o3d.io.read_triangle_mesh("../../Datasets/ModelNet40/chair/train/chair_0100.off")

np_array = data.convert_mesh_to_np_array(mesh)

plot_3d_voxels(np_array, voxel_size=1)
