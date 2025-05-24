import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhivyeshrk/diseases-and-symptoms-dataset")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file_path = '/root/.cache/kagglehub/datasets/dhivyeshrk/diseases-and-symptoms-dataset/versions/1/Final_Augmented_dataset_Diseases_and_Symptoms.csv'

df = pd.read_csv(file_path)
print(df.head())
df['diseases'].unique()[:200]
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
