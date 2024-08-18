import os
import time
import pandas as pd
import torch
from utils import data, helpers, models
from utils import essentials_text2model as et2m

DATA_ROOT_DIR = './Datasets/ModelNet40Cars'
CSV_FILE = 'dataset_clean.csv'

MODEL_SAVE_DIR = './Saved Weights/Text2Model'

DESCRIPTION_COLUMN = 'description'
FILE_PATH_COLUMN = 'file_name'

CLASS = 'car'

VOXEL_DIM = 64

PRE_LOAD = True
ADD_NOISE = True
EPS = 0.001
TRAIN_BATCH_SIZE = 8

LR = 0.001

LOG_INTERVAL = 200

NUM_EPOCHS = 200
SAVE_EVERY_EPOCH = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_file_path = os.path.join(DATA_ROOT_DIR, CSV_FILE)

df = pd.read_csv(csv_file_path)

train_df = df
# train_df, test_df = helpers.split_train_test(df)

embedding = data.EmbeddingT2M()
transform = data.load_off_to_tensor_custom(num_voxels_per_dim=VOXEL_DIM)

train_dataset = data.DataSet(
    df=train_df,
    root_dir=DATA_ROOT_DIR,
    descriptor=DESCRIPTION_COLUMN,
    file_path=FILE_PATH_COLUMN,
    embedding=embedding,
    transform=transform,
    pre_load=PRE_LOAD,
    add_noise=ADD_NOISE,
    eps=EPS
)
# test_dataset = data.DataSet3DGANs(
#     df=test_df,
#     root_dir=DATA_ROOT_DIR,
#     file_path=FILE_PATH_COLUMN,
#     transform=transform,
#     pre_load=PRE_LOAD
# )

train_loader = data.get_data_loader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
# test_loader = data.get_data_loader(test_dataset, batch_size=TEST_BATCH_SIZE)


model = models.Generator(out_dim=VOXEL_DIM, noise_dim=embedding.output_size).to(device)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch = epoch + 1
    print(f'======== Epoch {epoch} ========')

    t1 = time.time()
    print(f"Training Model...")
    disc_loss = et2m.train(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        log_interval=LOG_INTERVAL,
        device=device,
    )
    t2 = time.time()
    minutes, secs = helpers.get_time_elapsed(t1, t2)
    print(f"Total Model Loss: {round(disc_loss, 6)} Time Required: {minutes}m {round(secs, 2)}s")

    if epoch % SAVE_EVERY_EPOCH == 0:
        status = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(status, os.path.join(MODEL_SAVE_DIR, f'status{CLASS}_epoch{epoch}.pth'))
        print(f"Model saved to {MODEL_SAVE_DIR}")

    print()

print("Training Completed Successfully!")
