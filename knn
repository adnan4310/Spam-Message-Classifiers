import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
file_path = '/content/knn.csv'
classification_data = pd.read_csv(file_path, encoding='latin1')
classification_data = classification_data[['Category', 'Message']]
classification_data.columns = ['label', 'message']

# Encode labels
le = LabelEncoder()
classification_data['label'] = le.fit_transform(classification_data['label'])

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(classification_data['message'])
Y = classification_data['label']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_test)

# Evaluation
lr_accuracy = accuracy_score(Y_test, lr_predictions)
knn_accuracy = accuracy_score(Y_test, knn_predictions)

print("Logistic Regression Accuracy:", lr_accuracy)
print("KNN Accuracy:", knn_accuracy)
