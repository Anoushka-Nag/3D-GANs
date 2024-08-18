import os
import time
import pandas as pd
import torch
from utils import data, helpers, models
from utils import essentials_3dgans as e3d

DATA_ROOT_DIR = './Datasets/ModelNet40'
CSV_FILE = 'modelnet40_metadata_cleaned.csv'

MODEL_SAVE_DIR = './Saved Weights/3DGANs'

DESCRIPTION_COLUMN = 'class'
FILE_PATH_COLUMN = 'object_path'

CLASS = 'car'

VOXEL_DIM = 64
NOISE_SAMPLES = 2000
NOISE_DIM = 200

PRE_LOAD = True
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1000
NOISE_BATCH_SIZE = 16

LR_GENERATOR = 0.0025
LR_DISCRIMINATOR = 0.00001

LOG_INTERVAL_GENERATOR = 200
LOG_INTERVAL_DISCRIMINATOR = 200

NUM_EPOCHS = 30
SAVE_EVERY_EPOCH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_file_path = os.path.join(DATA_ROOT_DIR, CSV_FILE)

df = pd.read_csv(csv_file_path)
df = helpers.get_class_df(df, class_column=DESCRIPTION_COLUMN, class_name=CLASS)

train_df = df
# train_df, test_df = helpers.split_train_test(df)

transform = data.load_off_to_tensor_custom()

train_dataset = data.DataSet3DGANs(
    df=train_df,
    root_dir=DATA_ROOT_DIR,
    file_path=FILE_PATH_COLUMN,
    transform=transform,
    pre_load=PRE_LOAD
)
# test_dataset = data.DataSet3DGANs(
#     df=test_df,
#     root_dir=DATA_ROOT_DIR,
#     file_path=FILE_PATH_COLUMN,
#     transform=transform,
#     pre_load=PRE_LOAD
# )
noise_dataset = data.NoiseDataset(
    num_of_samples=NOISE_SAMPLES,
    noise_dim=NOISE_DIM
)

train_loader = data.get_data_loader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
# test_loader = data.get_data_loader(test_dataset, batch_size=TEST_BATCH_SIZE)
noise_loader = data.get_data_loader(noise_dataset, batch_size=NOISE_BATCH_SIZE)

noise_generator = data.noise_generator(NOISE_DIM)

model_generator = models.Generator(out_dim=VOXEL_DIM, noise_dim=NOISE_DIM).to(device)
model_discriminator = models.Discriminator(in_channels=1, dim=VOXEL_DIM).to(device)

criterion_generator = torch.nn.BCELoss()
criterion_discriminator = torch.nn.BCELoss()

optimizer_generator = torch.optim.Adam(model_generator.parameters(), lr=LR_GENERATOR)
optimizer_discriminator = torch.optim.Adam(model_discriminator.parameters(), lr=LR_DISCRIMINATOR)

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch = epoch + 1
    print(f'======== Epoch {epoch} ========')

    epochs_discriminator = (NUM_EPOCHS - epoch) // 10 + 1

    for i in range(epochs_discriminator):
        t1 = time.time()
        print(f"Training Discriminator...")
        disc_loss = e3d.train_discriminator(
            discriminator=model_discriminator,
            generator=model_generator,
            data_loader=train_loader,
            noise_generator=noise_generator,
            criterion_discriminator=criterion_discriminator,
            optimizer_discriminator=optimizer_discriminator,
            log_interval=LOG_INTERVAL_DISCRIMINATOR,
            device=device,
        )
        t2 = time.time()
        minutes, secs = helpers.get_time_elapsed(t1, t2)
        print(f"Total Discriminator Loss: {round(disc_loss, 6)} Time Required: {minutes}m {round(secs, 2)}s")

    epochs_generator = epoch // 10 + 1

    for i in range(epochs_generator):
        print(f"Training Generator...")
        t1 = time.time()
        gen_loss = e3d.train_generator(
            generator=model_generator,
            discriminator=model_discriminator,
            data_loader=noise_loader,
            criterion_generator=criterion_generator,
            optimizer_generator=optimizer_generator,
            log_interval=LOG_INTERVAL_GENERATOR,
            device=device
        )
        t2 = time.time()
        minutes, secs = helpers.get_time_elapsed(t1, t2)
        print(f"Total Generator Loss: {round(gen_loss, 6)} Time Required: {minutes}m {round(secs, 2)}s")

    epoch_end_time = time.time()
    minutes, secs = helpers.get_time_elapsed(start_time, epoch_end_time)
    print(f"Total Time Elapsed: {minutes}m {round(secs, 2)}s")

    if epoch % SAVE_EVERY_EPOCH == 0:
        status = {
            "generator_state_dict": model_generator.state_dict(),
            "discriminator_state_dict": model_discriminator.state_dict(),
            "optimizer_generator_state_dict": optimizer_generator.state_dict(),
            "optimizer_discriminator_state_dict": optimizer_discriminator.state_dict(),
        }

        torch.save(status, os.path.join(MODEL_SAVE_DIR, f'status{CLASS}_epoch{epoch}.pth'))
        print(f"Model saved to {MODEL_SAVE_DIR}")

    print()

print("Training Completed Successfully!")
