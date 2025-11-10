import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

file_path = '/content/drive/MyDrive/Ai Lab/Project/spam_Logistic_Regression_Googledataset.csv'
classification_data = pd.read_csv(file_path, encoding='latin1')

classification_data = classification_data[['v1', 'v2']]
classification_data.columns = ['label', 'message']

print("Dataset Preview:\n", classification_data.head())
print("Dataset Shape:", classification_data.shape)

X = classification_data['message']
Y = classification_data['label']

le = LabelEncoder()
Y = le.fit_transform(Y)

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lrc = LogisticRegression()
lrc.fit(X_train, Y_train)

predictions = lrc.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)
cm = confusion_matrix(Y_test, predictions)

print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy)

#sample_message = ["Congrats! you won car prize today. Claim your reward"]
sample_message = ["Have you comleted Ai project?"]
sample_vectorized = vectorizer.transform(sample_message)
sample_prediction = lrc.predict(sample_vectorized)
print("Sample Message:", sample_message[0])
print("Predicted Label:", le.inverse_transform(sample_prediction))
