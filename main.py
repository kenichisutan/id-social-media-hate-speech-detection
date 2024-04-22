import pandas as pd
from nltk.tokenize import word_tokenize

# Load dataset IDHSD_RIO_unbalanced_713_2017.txt
df = pd.read_csv('IDHSD_RIO_unbalanced_713_2017.txt', sep='\t', encoding='ISO-8859-1')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# ----------------------------
# PRE-PROCESSING
# ----------------------------
print("\nPre-processing the dataset...")

# Tokenization
df['tokenized_text'] = df['Tweet'].apply(word_tokenize)
# Lowercasing
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word.lower() for word in x])
# Remove non-alphanumeric tokens
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word.isalnum()])
# Remove stopwords
# TODO: Load Indonesian stopwords

# Display the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset after pre-processing:")
print(df.head())