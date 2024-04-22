import pandas as pd

# Load dataset IDHSD_RIO_unbalanced_713_2017.txt
df = pd.read_csv('IDHSD_RIO_unbalanced_713_2017.txt', sep='\t', encoding='ISO-8859-1')

# Display the first 5 rows of the dataset
print(df.head())