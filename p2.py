import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  


pd=df.head()
print("\nHead:\n",pd)
pd=df.shape
print("\nShape:", pd)
pd=df.isnull().sum()
print(pd)
pd= df.describe()
print("\nDescribe:\n", pd)
print()
pd= df.info()
print("\nInfo:\n", pd)

X= iris.data
y= iris.target
X_train, X_test, y_train, y_test = train_test_split(
 X , y,
test_size=0.2,
random_state=42,
stratify=y) 
     
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model= KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(f"k={k}, Accuracy={accuracy_score(y_test, preds):.3f}")