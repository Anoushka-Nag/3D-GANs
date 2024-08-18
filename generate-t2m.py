import torch
import glob
from utils import data, models, visualization

MODEL_FILE = './Saved Weights/Text2Model/statuscar_epoch200.pth'

DESCRIPTOR = 'Very powerful Dodge muscle car'

VOXEL_DIM = 64
NUM_OF_SAMPLES = 1
THRESHOLD = 0.5

embedding = data.EmbeddingT2M()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

status = torch.load(MODEL_FILE)

model = models.Generator(out_dim=VOXEL_DIM, noise_dim=embedding.output_size).to(device)
model.load_state_dict(status['model_state_dict'])

model.eval()

test_x = embedding(DESCRIPTOR)
test_x = test_x.squeeze().to(device)

out = model(test_x)
out = out.squeeze()
out = data.convert_probs_to_voxels(out, threshold=THRESHOLD)
voxels = out.data.cpu().numpy()

visualization.plot_3d_voxels(voxels, voxel_size=3)
