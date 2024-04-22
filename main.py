import pandas as pd
from nltk.tokenize import word_tokenize


# ----------------------------
# LOAD DATASET
# ----------------------------
print("Loading the dataset...")

# Dataset information:
'''
It consists of 713 tweets in Indonesian

Number of Non_HS tweets: 453
Number of HS tweets: 260.
Since this dataset is unbalanced, you might have to do over-sampling/down-sampling in order to create a balanced dataset.
'''

# Load dataset IDHSD_RIO_unbalanced_713_2017.txt
df = pd.read_csv('IDHSD_RIO_unbalanced_713_2017.txt', sep='\t', encoding='ISO-8859-1')
# Downsample the Non-HS tweets to 260
df_non_hs = df[df['Label'] == "Non_HS"].sample(n=260, random_state=42)
# Downsample the HS tweets to 260 (same amount)
df_hs = df[df['Label'] == "HS"].sample(n=260, random_state=42)
# Concatenate the downsampled Non-HS and HS tweets
df = pd.concat([df_non_hs, df_hs])
# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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
# Load Indonesian stopwords from stopwordbahasa.csv
stopwords = pd.read_csv('stopwordbahasa.csv')
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stopwords])

# Display the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset after pre-processing:")
print(df.head())

# ----------------------------
# DATA SPLITTING
# ----------------------------
print("\nSplitting the dataset into training and testing sets...")

# Split the dataset into training and testing sets (80% training, 20% testing)
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Display the shape of the training and testing sets
print("\nShape of the training set:", train_df.shape)
print("Shape of the testing set:", test_df.shape)