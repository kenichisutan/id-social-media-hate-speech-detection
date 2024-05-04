import os
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

def train():
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
    df_non_hs = df[ df[ 'Label' ] == "Non_HS" ].sample(n=260, random_state=42)
    # Downsample the HS tweets to 260 (same amount)
    df_hs = df[ df[ 'Label' ] == "HS" ].sample(n=260, random_state=42)
    # Concatenate the downsampled Non-HS and HS tweets
    df = pd.concat([ df_non_hs, df_hs ])
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
    df[ 'tokenized_text' ] = df[ 'Tweet' ].apply(word_tokenize)
    print("\nTokenized text:\n", df[ 'tokenized_text' ].head())
    # If the current token is a #, remove the current token and the next token
    df[ 'tokenized_text' ] = df[ 'tokenized_text' ].apply(lambda x: [ x[i] for i in range(len(x)) if x[i] != '#' and (i == 0 or x[i-1] != '#') ])
    # If the current token is a @, remove the current token and the next token
    df[ 'tokenized_text' ] = df[ 'tokenized_text' ].apply(lambda x: [ x[i] for i in range(len(x)) if x[i] != '@' and (i == 0 or x[i-1] != '@') ])
    # Lowercasing
    df[ 'tokenized_text' ] = df[ 'tokenized_text' ].apply(lambda x: [ word.lower() for word in x ])
    print("\nLowercased text:\n", df[ 'tokenized_text' ].head())
    # # Remove non-alphanumeric tokens
    # df[ 'tokenized_text' ] = df[ 'tokenized_text' ].apply(lambda x: [ word for word in x if word.isalnum() ])
    # print("\nText with alphanumeric tokens:\n", df[ 'tokenized_text' ].head())
    # Remove tokens with only non-alphanumeric characters
    df[ 'tokenized_text' ] = df[ 'tokenized_text' ].apply(lambda x: [ word for word in x if any(char.isalnum() for char in word) ])
    print("\nText with alphanumeric tokens:\n", df[ 'tokenized_text' ].head())
    # Remove stopwords
    # Load Indonesian stopwords from stopwordbahasa.csv
    stopwords = pd.read_csv('stopwordbahasa.csv')
    df[ 'tokenized_text' ] = df[ 'tokenized_text' ].apply(lambda x: [ word for word in x if word not in stopwords ])
    print("\nText without stopwords:\n", df[ 'tokenized_text' ].head())

    # Display the first 5 rows of the dataset
    print("\nFirst 5 rows of the dataset after pre-processing:")
    print(df.head())

    # ----------------------------
    # DATA SPLITTING
    # ----------------------------
    print("\nSplitting the dataset into training and testing sets...")

    # Split the dataset into training and testing sets (80% training, 20% testing)
    # Randomly shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the dataset
    train_size = int(0.8 * len(df))
    train_df = df[ :train_size ]
    test_df = df[ train_size: ]

    # Display the shape of the training and testing sets
    print("\nShape of the training set:", train_df.shape)
    print("Shape of the testing set:", test_df.shape)

    # ----------------------------
    # FEATURE EXTRACTION
    # ----------------------------
    print("\nExtracting features from the dataset...")

    # Calculate the term frequency values for words in the training set
    train_tf = {}
    train_tf[ 'HS' ] = {}
    train_tf[ 'Non_HS' ] = {}
    for index, row in train_df.iterrows():
        for word in row[ 'tokenized_text' ]:
            if word not in train_tf[ row[ 'Label' ] ]:
                train_tf[ row[ 'Label' ]][ word ] = 1
            else:
                train_tf[ row[ 'Label' ]][ word ] += 1

    # Display the term frequency values for the training set
    print("\nTerm frequency values for the training set:")
    print(train_tf[ 'HS' ])
    print(train_tf[ 'Non_HS' ])

    # ----------------------------
    # VECTORIZE TEXT DATA
    # ----------------------------
    print("\nVectorizing text data...")

    # Combine tokenized text back into strings for vectorization
    train_df['clean_text'] = train_df['tokenized_text'].apply(lambda x: ' '.join(x))
    test_df['clean_text'] = test_df['tokenized_text'].apply(lambda x: ' '.join(x))

    # Create CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['Label']

    # Transform the testing data
    X_test = vectorizer.transform(test_df['clean_text'])
    y_test = test_df['Label']

    print("\nShape of the training data:", X_train.shape)
    print("Shape of the testing data:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    # ----------------------------
    # CLASSIFICATION
    # ----------------------------
    print("\nTraining the classification model...")

    # Initialize Logistic Regression model
    clf = LogisticRegression(max_iter=10000, random_state=42)

    # Train model
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    return y_pred, y_test, test_df, vectorizer, clf

def eval(y_pred, y_test):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='HS')
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-score: {f1_score*100:.2f}%")

    return accuracy, precision, recall, f1_score

iterations = 100
avg_accuracy, avg_precision, avg_recall, avg_f1_score = 0, 0, 0, 0
y_test, y_pred = None, None
for i in range(iterations):
    print(f"\nIteration {i+1}:")
    # Train the model
    y_pred, y_test, test_df, vectorizer, clf = train()
    # Evaluate the model
    accuracy, precision, recall, f1_score = eval(y_pred, y_test)
    avg_accuracy += accuracy
    avg_precision += precision
    avg_recall += recall
    avg_f1_score += f1_score

# Calculate average accuracy, precision, recall, and F1-score
avg_accuracy /= iterations
avg_precision /= iterations
avg_recall /= iterations
avg_f1_score /= iterations

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non_HS', 'HS'], yticklabels=['Non_HS', 'HS'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("\nAverage metrics:")
print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
print(f"Average Precision: {avg_precision*100:.2f}%")
print(f"Average Recall: {avg_recall*100:.2f}%")
print(f"Average F1-score: {avg_f1_score*100:.2f}%")
