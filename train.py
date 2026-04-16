import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Dense, Input

# Load dataset
data = pd.read_csv(r"c:\Users\ASUS\Downloads\archive (3)\Spam_SMS.csv")

# Fix column names
if 'Class' in data.columns:
    data = data[['Class', 'Message']]
elif 'v1' in data.columns:
    data = data[['v1', 'v2']]

data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Model
model = Sequential([
    Input(shape=(X_train_vec.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_vec, y_train, epochs=5, batch_size=32)

# Save
model.save("model.keras")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Done!")