import os
import time
import pandas as pd
from utils import data

LOG_INTERVAL = 500

ROOT_DIR = '../../Datasets/ModelNet40'
FILE_NAME = 'modelnet40_metadata.csv'
CLEANED_FILE_NAME = 'modelnet40_metadata_cleaned.csv'
FILE_PATH_COLUMN = "object_path"

csv_file_path = os.path.join(ROOT_DIR, FILE_NAME)
cleaned_file_path = os.path.join(ROOT_DIR, CLEANED_FILE_NAME)

df = pd.read_csv(csv_file_path)

idx = pd.Series([True] * len(df))

start_time = time.time()
for i, row in df.iterrows():
    try:
        path = os.path.join(ROOT_DIR, row[FILE_PATH_COLUMN])
        tensor = data.load_off_to_tensor(path)
    except Exception as e:
        idx[i] = False

    if i % LOG_INTERVAL == 0:
        current_time = time.time()
        elapsed_time = current_time - start_time
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        print(f"Finished checking {i}/{len(df)} ({round(i / len(df) * 100, 2)}%) in {minutes} minutes and {round(seconds, 2)} seconds")


current_time = time.time()
elapsed_time = current_time - start_time
minutes = elapsed_time // 60
seconds = elapsed_time % 60
print(f"Finished checking {len(df)}/{len(df)} (100.0%) in {minutes} minutes and {round(seconds, 2)} seconds")


cleaned = df[idx]
cleaned.to_csv(cleaned_file_path, index=False)
