import plotly.graph_objects as go
import numpy as np


def plot_3d_voxels(voxel_data: np.ndarray, voxel_size: int = 1) -> None:
    x, y, z = np.where(voxel_data == 1)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=voxel_size,
                    symbol='square',
                    color='blue'
                )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )

    fig.show()
