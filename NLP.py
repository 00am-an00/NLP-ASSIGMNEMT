import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Step 1: Download and extract the dataset
url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
dataset_dir = "rt-polaritydata"
tar_gz_file = "rt-polaritydata.tar.gz"

# Download dataset
urllib.request.urlretrieve(url, tar_gz_file)

# Extract the tar.gz file
with tarfile.open(tar_gz_file, "r:gz") as tar:
    tar.extractall()

# Step 2: Load data
positive_file = os.path.join(dataset_dir, 'rt-polarity.pos')
negative_file = os.path.join(dataset_dir, 'rt-polarity.neg')

# Load positive and negative reviews
with open(positive_file, 'r', encoding='latin-1') as f:
    positive_reviews = f.readlines()

with open(negative_file, 'r', encoding='latin-1') as f:
    negative_reviews = f.readlines()

# Create labels: 1 for positive, 0 for negative
positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

# Combine data
reviews = positive_reviews + negative_reviews
labels = positive_labels + negative_labels

# Convert to DataFrame
df = pd.DataFrame({'review': reviews, 'label': labels})

# Step 3: Data splitting (train, validation, test sets)
# Train: 4,000 positive and 4,000 negative, Validation: 500 each, Test: 831 each
train_pos, val_pos, test_pos = positive_reviews[:4000], positive_reviews[4000:4500], positive_reviews[4500:]
train_neg, val_neg, test_neg = negative_reviews[:4000], negative_reviews[4000:4500], negative_reviews[4500:]

# Create training, validation, and test sets
train_data = train_pos + train_neg
train_labels = [1] * len(train_pos) + [0] * len(train_neg)

val_data = val_pos + val_neg
val_labels = [1] * len(val_pos) + [0] * len(val_neg)

test_data = test_pos + test_neg
test_labels = [1] * len(test_pos) + [0] * len(test_neg)

# Step 4: Preprocessing (TF-IDF vectorization)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data)
X_val = vectorizer.transform(val_data)
X_test = vectorizer.transform(test_data)

# Step 5: Model training (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# Step 6: Evaluation
# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
tp, fn, fp, tn = confusion_matrix(test_labels, y_pred).ravel()
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)

# Display metrics
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: Save model and vectorizer
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
