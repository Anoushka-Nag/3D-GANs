import torch
from utils import data, models, visualization

MODEL_FILE = './Saved Weights/3DGANs/statuscar_epoch30.pth'

CLASS = 'chair'

VOXEL_DIM = 64
NOISE_DIM = 200
NUM_OF_SAMPLES = 1
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_generator = data.noise_generator(noise_dim=NOISE_DIM)

status = torch.load(MODEL_FILE)

generator = models.Generator(out_dim=VOXEL_DIM, noise_dim=NOISE_DIM).to(device)
generator.load_state_dict(status['generator_state_dict'])

discriminator = models.Discriminator(in_channels=1, dim=VOXEL_DIM).to(device)
discriminator.load_state_dict(status['discriminator_state_dict'])

generator.eval()
discriminator.eval()

fake_x = noise_generator(1).to(device)
out = generator(fake_x)
score = discriminator(out).item()

print(f'Score from discriminator: {round(score, 2)}')

out = out.squeeze()
out = data.convert_probs_to_voxels(out, threshold=THRESHOLD)
voxels = out.data.cpu().numpy()

visualization.plot_3d_voxels(voxels, voxel_size=1)
