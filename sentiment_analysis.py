import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Read CSV file with a specific encoding
try:
    data = pd.read_csv('./test.csv', encoding='latin1')  # Try 'ISO-8859-1' if 'latin1’ doesn’t work
except Exception as e:
    print(f"Error reading CSV file: {e}")
    raise

# Print the DataFrame
print(data.head())

# Check if 'text' and 'sentiment' columns exist and fill missing values
if 'text' in data.columns:
    data['text'] = data['text'].fillna('')
else:
    print("The 'text' column is missing from the CSV file.")
    raise KeyError("The 'text' column is missing from the CSV file.")

if 'sentiment' in data.columns:
    data['sentiment'] = data['sentiment'].fillna('neutral')  # Assuming 'neutral' is a default sentiment
else:
    print("The 'sentiment' column is missing from the CSV file.")
    raise KeyError("The 'sentiment' column is missing from the CSV file.")

# Drop rows where sentiment labels are NaN
data = data.dropna(subset=['sentiment'])

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to clean the text
def preprocess_text(text):
    try:
        # Convert text to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize the text
        words = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        cleaned_text = [stemmer.stem(word) for word in words if word not in stop_words]
        
        return ' '.join(cleaned_text)
    except Exception as e:
        print(f"Error processing text: {e}")
        return text

# Apply preprocessing to the dataset's text column
data['cleaned_text'] = data['text'].apply(preprocess_text)

# View the cleaned text
print(data['cleaned_text'].head())

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the cleaned text data
X = vectorizer.fit_transform(data['cleaned_text'])

# Map sentiment labels to numerical values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
data['sentiment_label'] = data['sentiment'].map(sentiment_mapping)

# Check for any missing values in sentiment_label
print("Missing values in sentiment_label after mapping:")
print(data['sentiment_label'].isnull().sum())

# Prepare target variable
y = data['sentiment_label']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))



#new predictions:
# Example new text data
new_texts = [
    "I love this product! It’s amazing.",
    "I’m not sure how I feel about this.",
    "This is the worst experience I’ve ever had.",
    "I am very happy",
    " I don't think it's a good decision"
]

# Preprocess the new texts
new_texts_cleaned = [preprocess_text(text) for text in new_texts]

# Transform the new texts using the fitted TF-IDF vectorizer
new_X = vectorizer.transform(new_texts_cleaned)

# Make predictions
new_predictions = model.predict(new_X)

# Convert numerical predictions back to sentiment labels
inverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
new_predictions_labels = [inverse_sentiment_mapping[pred] for pred in new_predictions]

# Print predictions
for text, sentiment in zip(new_texts, new_predictions_labels):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

#saving model
import joblib

# Save the model
joblib.dump(model, 'sentiment_model.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
