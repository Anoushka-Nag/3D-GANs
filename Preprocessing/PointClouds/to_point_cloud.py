import open3d as o3d
import open3d.core as o3c
import numpy as np


mesh = o3d.io.read_triangle_mesh("../../Datasets/airplane_0627.off")
mesh.scale(64 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
pcd = mesh.sample_points_uniformly(50000)
points = o3c.Tensor(pcd.points)
pcd = o3d.geometry.PointCloud(points)
o3d.visualization.draw_geometries([pcd])
