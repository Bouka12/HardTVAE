import pandas as pd
from utils import get_data

dataset_name = "Hypothyroid" # "Hepatitis"     # "HeartCleveland" #"BCWDD"
df = get_data(dataset_name=dataset_name)
print(df.head())
print(df['Outcome'].value_counts())