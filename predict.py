
import joblib

# Load saved model
model = joblib.load("model/spam_model.pkl")

# Load saved TF-IDF vectorizer
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# Get a new SMS message from the user
message = input("Enter your SMS message: ")

# Convert message into numerical features
message_tfidf = tfidf.transform([message])

# Predict the message
prediction = model.predict(message_tfidf)

# Display the result
print("\nPrediction:", prediction[0].upper())
