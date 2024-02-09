# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

# Load Sentiment140 dataset
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None, names=columns)

# Drop unnecessary columns
df = df.drop(['id', 'date', 'flag', 'user'], axis=1)

# Map target labels (0 for negative, 4 for positive) to (0 for negative, 1 for positive)
df['target'] = df['target'].map({0: 0, 4: 1})

# Feature engineering (you can expand this based on your project)
# Here, a simple bag-of-words approach is used with CountVectorizer
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

joblib.dump(model, 'your_model.pkl')
