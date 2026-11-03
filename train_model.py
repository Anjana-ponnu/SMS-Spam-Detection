import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Remove duplicate messages
data = data.drop_duplicates()

# Separate input and output
X = data['message']
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical features using TF-IDF

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)



# Create Naive Bayes model

model = MultinomialNB(alpha=0.1)

# Train the model

model.fit(X_train_tfidf, y_train)

print("\nModel Training Completed Successfully!")

# Predict testing data

y_pred = model.predict(X_test_tfidf)

print("\nModel Predictions:")
print(y_pred[:10])

# Calculate model accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy * 100, "%")


# Classification report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model

joblib.dump(model, "model/spam_model.pkl")

# Save TF-IDF vectorizer

joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel and Vectorizer Saved Successfully!")