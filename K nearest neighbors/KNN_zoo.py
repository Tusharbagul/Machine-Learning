import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#dataset loading
pd.set_option(
          'display.max_columns', 100
          )
#loading dataset
df = pd.read_csv(
          'C:/Users/Tushar Bagul/Desktop/Data_Science/Assignments/KNN/Zoo.csv'
          )
df.head()
df.info()
df.isnull().sum()
df['type'].value_counts()
df.columns

#defining X and y
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

#train test split
X_train,X_test, y_train,y_test = train_test_split(
          X, y, test_size=0.3, random_state=455
          )
#initializing knn
knn = KNeighborsClassifier(
          n_neighbors=5, p=2, weights='distance'
          )
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
y_pred = knn.predict(X_test)
y_pred

#report
print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))













