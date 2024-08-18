import os
import pandas as pd


DATA_ROOT_DIR = '../../Datasets/ModelNet40'
CSV_FILE = 'modelnet40_metadata_cleaned.csv'

DESCRIPTION_COLUMN = 'class'


csv_file_path = os.path.join(DATA_ROOT_DIR, CSV_FILE)

df = pd.read_csv(csv_file_path)

for class_label, df_class in df.groupby(DESCRIPTION_COLUMN):
    print(class_label, ': ', len(df_class))
