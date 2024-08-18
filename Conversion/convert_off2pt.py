import os
import time
import pandas as pd
import torch
from utils import data


DATA_ROOT_DIR = '../Datasets/ModelNet40'
CSV_FILE = 'modelnet40_metadata_cleaned.csv'

DATA_SAVE_DIR = './Datasets/ModelNet40Converted'

DESCRIPTION_COLUMN = 'class'
FILE_PATH_COLUMN = 'object_path'

VOXEL_DIM = 64

csv_file_path = os.path.join(DATA_ROOT_DIR, CSV_FILE)

df = pd.read_csv(csv_file_path)

transform = data.load_off_to_tensor_custom()

for i, row in df.iterrows():
    print(row[FILE_PATH_COLUMN][:-4])
    break
    
