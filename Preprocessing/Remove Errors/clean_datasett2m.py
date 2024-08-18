import pandas as pd

EXCEL_PATH = '../../Datasets/ModelNet40Cars/dataset.xlsx'
SAVE_PATH = '../../Datasets/ModelNet40Cars/dataset_clean.csv'

df = pd.read_excel(EXCEL_PATH)

cleaned_df = df.dropna()
cleaned_df = pd.DataFrame(
    {
        'file_name': cleaned_df['file_name '].map(lambda x: x + '.off'),
        'description': cleaned_df['new desc']
    }
)

cleaned_df.to_csv(SAVE_PATH, index=False)
