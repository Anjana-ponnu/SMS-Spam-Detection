from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Keep negation words because they can change message meaning
stop_words = stop_words - {'no', 'not', 'nor'}

stemmer = PorterStemmer()



def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and perform stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    # Join words into a sentence
    return " ".join(words)


# Load the SMS dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display first five records
print(data.head())

# Total number of messages
print("\nDataset Shape:")
print(data.shape)

# Dataset information
print("\nDataset Information:")
data.info()

# Count Spam and Ham messages
print("\nMessage Distribution:")
print(data["label"].value_counts())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check duplicate messages
print("\nDuplicate Messages:")
print(data.duplicated().sum())

# Remove duplicate messages
data = data.drop_duplicates()

# Check updated dataset
print("\nDataset after removing duplicates:")
print(data.shape)

# Check duplicates again
print("\nRemaining duplicates:")
print(data.duplicated().sum())

# Apply text preprocessing
data['clean_message'] = data['message'].apply(preprocess_text)

# Display original and cleaned messages
print("\nOriginal and Cleaned Messages:")

print(data[['message', 'clean_message']].head(10))


# Convert text into numerical features

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data['clean_message'])

print("\nTF-IDF Shape:")
print(X.shape)

y = data['label']

print("\nInput Shape:")
print(X.shape)

print("\nOutput Shape:")
print(y.shape)