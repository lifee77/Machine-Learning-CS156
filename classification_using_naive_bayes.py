import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/milindsoorya/Spam-Classifier-in-python/main/dataset/spam.csv", encoding='latin-1')
labels, texts = df['v1'], df['v2']

# Convert labels to binary (spam = 1, ham = 0)
labels = (labels == "spam").astype(int)

# Create TfidfVectorizer and transform text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Extract feature probabilities
feature_log_probs = model.feature_log_prob_
p_X_given_spam = np.exp(feature_log_probs[1])  # P(word | spam)
p_X_given_ham = np.exp(feature_log_probs[0])   # P(word | ham)

# Get feature names (words)
feature_names = np.array(vectorizer.get_feature_names_out())

# Select the top 30 spam-related and ham-related words
top_30_spam_indices = np.argsort(p_X_given_spam)[-30:][::-1]
top_30_ham_indices = np.argsort(p_X_given_ham)[-30:][::-1]

top_30_spam_words = feature_names[top_30_spam_indices]
top_30_spam_probs = p_X_given_spam[top_30_spam_indices]

top_30_ham_words = feature_names[top_30_ham_indices]
top_30_ham_probs = p_X_given_ham[top_30_ham_indices]

# Plot histogram for top spam words
plt.figure(figsize=(12, 6))
sns.barplot(x=top_30_spam_words, y=top_30_spam_probs, color="red", alpha=0.6)
plt.xticks(rotation=90)
plt.xlabel("Words")
plt.ylabel("P(Word | Spam)")
plt.title("Top 30 Words with Highest P(Word | Spam)")
plt.show()

# Plot histogram for top ham words
plt.figure(figsize=(12, 6))
sns.barplot(x=top_30_ham_words, y=top_30_ham_probs, color="blue", alpha=0.6)
plt.xticks(rotation=90)
plt.xlabel("Words")
plt.ylabel("P(Word | Ham)")
plt.title("Top 30 Words with Highest P(Word | Ham)")
plt.show()
