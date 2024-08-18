import os
import time
import pandas as pd
import torch
from utils import data, helpers, models
from utils import essentials_cgans as ecg

DATA_ROOT_DIR = './Datasets/ModelNet40'
CSV_FILE = 'modelnet40_metadata_cleaned.csv'

MODEL_SAVE_DIR = './Saved Weights/CGANs'

DESCRIPTION_COLUMN = 'class'
FILE_PATH_COLUMN = 'object_path'

VOXEL_DIM = 64
NOISE_SAMPLES = 8000
NOISE_DIM = 200

PRE_LOAD = False
TRAIN_BATCH_SIZE = 32
NOISE_BATCH_SIZE = 32

LR_GENERATOR = 0.0025
LR_DISCRIMINATOR = 0.00001

LOG_INTERVAL_GENERATOR = 50
LOG_INTERVAL_DISCRIMINATOR = 50

NUM_EPOCHS = 10
SAVE_EVERY_EPOCH = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_file_path = os.path.join(DATA_ROOT_DIR, CSV_FILE)

df = pd.read_csv(csv_file_path)
classes = helpers.get_uniques(df, DESCRIPTION_COLUMN)

embedding = data.EmbeddingText()
transform = data.load_off_to_tensor_custom()

train_dataset = data.DataSetCGANs(
    df=df,
    root_dir=DATA_ROOT_DIR,
    descriptor=DESCRIPTION_COLUMN,
    file_path=FILE_PATH_COLUMN,
    embedding=embedding,
    transform=transform,
    pre_load=PRE_LOAD
)

noise_dataset = data.DatasetCGANsGenerator(
    num_of_samples=NOISE_SAMPLES,
    classes=classes,
    embedding=embedding,
    noise_dim=NOISE_DIM
)

train_loader = data.get_data_loader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
noise_loader = data.get_data_loader(noise_dataset, batch_size=NOISE_BATCH_SIZE)

noise_generator = data.noise_generator(NOISE_DIM)

model_generator = models.GeneratorCGANs(out_dim=VOXEL_DIM, noise_dim=NOISE_DIM, embedding_dim=embedding.output_size).to(device)
model_discriminator = models.DiscriminatorCGANs(in_channels=1, embedding_dim=embedding.output_size, dim=VOXEL_DIM).to(device)

criterion_generator = torch.nn.BCELoss()
criterion_discriminator = torch.nn.BCELoss()

optimizer_generator = torch.optim.Adam(model_generator.parameters(), lr=LR_GENERATOR)
optimizer_discriminator = torch.optim.Adam(model_discriminator.parameters(), lr=LR_DISCRIMINATOR)

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch = epoch + 1
    print(f'======== Epoch {epoch} ========')

    epochs_discriminator = (NUM_EPOCHS - epoch) // 5 + 1
    # for i in range(epochs_discriminator):
    #     t1 = time.time()
    #     print(f"Training Discriminator...")
    #     disc_loss = ecg.train_discriminator(
    #         discriminator=model_discriminator,
    #         generator=model_generator,
    #         data_loader=train_loader,
    #         noise_generator=noise_generator,
    #         criterion_discriminator=criterion_discriminator,
    #         optimizer_discriminator=optimizer_discriminator,
    #         log_interval=LOG_INTERVAL_DISCRIMINATOR,
    #         device=device,
    #     )
    #     t2 = time.time()
    #     minutes, secs = helpers.get_time_elapsed(t1, t2)
    #     print(f"Total Discriminator Loss: {round(disc_loss, 6)} Time Required: {minutes}m {round(secs, 2)}s")

    epochs_generator = (epoch-1) // 5 + 1
    for i in range(epochs_generator):
        print(f"Training Generator...")
        t1 = time.time()
        gen_loss = ecg.train_generator(
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

        torch.save(status, os.path.join(MODEL_SAVE_DIR, f'status_epoch{epoch}.pth'))
        print(f"Model saved to {MODEL_SAVE_DIR}")

    print()

print("Training Completed Successfully!")
