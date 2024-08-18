import os
import random
import warnings
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("error")


def load_off(file_path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(file_path)
    return mesh


def convert_mesh_to_voxel_grid(
        mesh: o3d.geometry,
        num_voxels_per_dim: int = 64,
        voxel_size: int = 1
) -> o3d.geometry.VoxelGrid:
    mesh.scale(
        (num_voxels_per_dim - 1) / (np.max(mesh.get_max_bound() - mesh.get_min_bound())),
        center=mesh.get_center()
    )
    mesh.compute_vertex_normals()

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=voxel_size
    )

    return voxel_grid


def convert_voxel_to_np_array(
        voxel_grid: o3d.geometry.VoxelGrid,
        num_voxels_per_dim: int = 64
) -> np.ndarray:
    np_array = np.zeros((num_voxels_per_dim, num_voxels_per_dim, num_voxels_per_dim))

    for voxel in voxel_grid.get_voxels():
        grid = voxel.grid_index
        np_array[grid[0], grid[1], grid[2]] = 1

    return np_array


def convert_mesh_to_np_array(
        mesh: o3d.geometry.VoxelGrid,
        num_voxels_per_dim: int = 64,
        voxel_size: int = 1
) -> np.ndarray:
    voxel_grid = convert_mesh_to_voxel_grid(mesh, num_voxels_per_dim, voxel_size)
    np_array = convert_voxel_to_np_array(voxel_grid, num_voxels_per_dim)
    return np_array


def load_off_to_tensor(
        file_path: str,
        num_voxels_per_dim: int = 64,
        voxel_size: int = 1
) -> torch.Tensor:
    mesh = load_off(file_path)
    np_array = convert_mesh_to_np_array(mesh, num_voxels_per_dim, voxel_size)
    return torch.from_numpy(np_array).to(torch.float32).unsqueeze(0)


def load_off_to_tensor_custom(
        num_voxels_per_dim: int = 64,
        voxel_size: int = 1,
):
    return lambda x: load_off_to_tensor(x, num_voxels_per_dim, voxel_size)


def convert_probs_to_voxels(
        probs: torch.Tensor,
        threshold: float = 0.5
) -> torch.Tensor:
    out = (probs > threshold).float()
    return out


def get_data_loader(
        dataset: data.Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
) -> data.DataLoader:
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader


class Embedding3DGANs:
    def __init__(self, target: str):
        self.target = target

    def __call__(self, x: str) -> torch.Tensor:
        val = 1 if self.target == x else 0
        return torch.tensor([val]).to(torch.float32)


class EmbeddingText:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        test = self.model.encode('test')
        self.output_size = test.shape[0]

    def __call__(self, x):
        out = self.model.encode(x)
        out = out.reshape(1, -1)
        out = torch.tensor(out).to(torch.float32)
        return out


class DataSet(data.Dataset):
    data_set = None

    def __init__(
            self,
            df: pd.DataFrame,
            root_dir: str,
            descriptor: str = "description",
            file_path: str = "file_name",
            embedding=None,
            transform=None,
            pre_load: bool = False,
            add_noise: bool = False,
            eps: float = 0.001
    ):
        super(DataSet, self).__init__()

        self.df = df
        self.root_dir = root_dir
        self.file_path = file_path
        self.descriptor = descriptor
        self.embedding = embedding
        self.transform = transform
        self.pre_load = pre_load
        self.add_noise = add_noise
        self.eps = eps

        if self.pre_load:
            self.load_to_memory()

    def __len__(self):
        return len(self.df)

    def get_item(self, idx: int):
        row = self.df.iloc[idx]
        file_name = row[self.file_path]
        descriptor = row[self.descriptor]
        file_path = os.path.join(self.root_dir, file_name)

        x = self.embedding(descriptor) if self.embedding is not None else descriptor
        y = self.transform(file_path) if self.transform is not None else file_path

        return x, y

    def load_to_memory(self):
        self.data_set = []
        for idx in range(len(self.df)):
            self.data_set.append(self.get_item(idx))

    def __getitem__(self, idx: int):
        x, y = self.data_set[idx] if self.pre_load else self.get_item(idx)

        if self.add_noise:
            x = x + torch.rand(x.shape) * self.eps

        return x, y


class DataSet3DGANs(data.Dataset):
    data_set = None

    def __init__(
            self,
            df: pd.DataFrame,
            root_dir: str,
            file_path: str = "file_name",
            transform=None,
            pre_load: bool = False
    ):
        super(DataSet3DGANs, self).__init__()

        self.df = df
        self.root_dir = root_dir
        self.file_path = file_path
        self.transform = transform
        self.pre_load = pre_load

        if self.pre_load:
            self.load_to_memory()

    def __len__(self):
        return len(self.df)

    def get_item(self, idx: int):
        row = self.df.iloc[idx]
        file_name = row[self.file_path]
        file_path = os.path.join(self.root_dir, file_name)

        x = self.transform(file_path) if self.transform is not None else file_path

        return x

    def load_to_memory(self):
        self.data_set = []
        for idx in range(len(self.df)):
            self.data_set.append(self.get_item(idx))

    def __getitem__(self, idx: int):
        if self.pre_load:
            return self.data_set[idx]
        else:
            return self.get_item(idx)


class DataSetCGANs(data.Dataset):
    data_set = None

    def __init__(
            self,
            df: pd.DataFrame,
            root_dir: str,
            descriptor: str = "class",
            file_path: str = "file_name",
            embedding=None,
            transform=None,
            pre_load: bool = False
    ):
        super(DataSetCGANs, self).__init__()

        self.df = df
        self.root_dir = root_dir
        self.descriptor = descriptor
        self.file_path = file_path
        self.transform = transform
        self.embedding = embedding
        self.pre_load = pre_load
        if self.pre_load:
            self.load_to_memory()

    def __len__(self):
        return len(self.df)

    def get_item(self, idx: int):
        row = self.df.iloc[idx]
        description = row[self.descriptor]
        file_name = row[self.file_path]
        file_path = os.path.join(self.root_dir, file_name)

        if self.embedding:
            x = self.embedding(description) if self.embedding is not None else description
            x = x.reshape(1, -1)
            x = x.to(torch.float32)
        else:
            x = description

        y = self.transform(file_path) if self.transform is not None else file_path

        return x, y

    def load_to_memory(self):
        self.data_set = []
        for idx in range(len(self.df)):
            self.data_set.append(self.get_item(idx))

    def __getitem__(self, idx: int):
        if self.pre_load:
            return self.data_set[idx]
        else:
            return self.get_item(idx)


def noise_generator(noise_dim: int):
    return lambda num_of_samples: torch.rand((num_of_samples, 1, noise_dim))


class NoiseDataset(data.Dataset):
    def __init__(self, num_of_samples: int, noise_dim: int = 200):
        self.num_of_samples = num_of_samples
        self.noise_dim = noise_dim

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return torch.rand(1, self.noise_dim)


class DatasetCGANsGenerator(data.Dataset):
    embeddings = []

    def __init__(self, num_of_samples: int, classes, embedding=None, noise_dim: int = 200, pre_load: bool = True):
        self.num_of_samples = num_of_samples
        self.classes = classes
        self.embedding = embedding
        self.noise_dim = noise_dim
        self.pre_load = pre_load
        if self.pre_load:
            self.load_to_memory()

    def get_embedding(self, idx):
        embedding = self.embedding(self.classes[idx])
        embedding = embedding.reshape(1, -1)
        embedding = embedding.to(torch.float32)
        return embedding

    def load_to_memory(self):
        self.embeddings = []
        for i in range(len(self.classes)):
            self.embeddings.append(self.get_embedding(i))

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        class_idx = random.randint(0, len(self.classes) - 1)

        if self.pre_load:
            embedding = self.embeddings[class_idx]
        else:
            embedding = self.get_embedding(self.classes[class_idx])

        noise = torch.rand(1, self.noise_dim)
        return embedding, noise
